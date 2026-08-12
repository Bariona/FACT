"""TCP inference server/client with torch/pickle-serialized messages.

Security note: payloads are deserialized with pickle (via ``torch.load``),
which can execute arbitrary code. Only bind the server to localhost or a
trusted network, and only connect the client to servers you control.
"""

from __future__ import annotations

import socket
import pickle
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable

import torch


def _set_tcp_socket_options(sock: socket.socket) -> None:
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        pass


class TorchSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes):
        buffer = BytesIO(data)
        return torch.load(buffer, map_location="cpu", weights_only=False)


class PickleSerializer:
    PREFIX = b"FACT_PICKLE_V1\0"

    @classmethod
    def is_payload(cls, data: bytes) -> bool:
        return data.startswith(cls.PREFIX)

    @classmethod
    def to_bytes(cls, data: Any) -> bytes:
        return cls.PREFIX + pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, data: bytes):
        if not cls.is_payload(data):
            raise ValueError("not a FACT pickle payload")
        return pickle.loads(data[len(cls.PREFIX) :])


def _to_pickle_safe(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, dict):
        return {k: _to_pickle_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_pickle_safe(v) for v in data]
    if isinstance(data, tuple):
        return tuple(_to_pickle_safe(v) for v in data)
    return data


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseTcpInferenceServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, backlog: int = 1, timeout_s: float = 1.0):
        self.running = True
        self.host = host
        self.port = int(port)
        self.backlog = int(backlog)
        self.timeout_s = float(timeout_s)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.backlog)
        self.server_socket.settimeout(self.timeout_s)
        self._endpoints: dict[str, EndpointHandler] = {}
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        self.running = False
        return {"status": "ok", "message": "Server stopping"}

    def _handle_ping(self) -> dict[str, Any]:
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True) -> None:
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    @staticmethod
    def _recv_exact(conn: socket.socket, size: int) -> bytes:
        chunks = []
        received = 0
        while received < size:
            chunk = conn.recv(size - received)
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _recv_message(self, conn: socket.socket) -> bytes:
        size = int.from_bytes(self._recv_exact(conn, 4), "big")
        return self._recv_exact(conn, size)

    @staticmethod
    def _send_message(conn: socket.socket, payload: bytes) -> None:
        conn.sendall(len(payload).to_bytes(4, "big"))
        conn.sendall(payload)

    def _serve_connection(self, conn: socket.socket) -> None:
        with conn:
            while self.running:
                try:
                    message = self._recv_message(conn)
                except (ConnectionError, OSError):
                    break
                try:
                    t_decode = time.perf_counter()
                    if PickleSerializer.is_payload(message):
                        serializer = PickleSerializer
                        request = PickleSerializer.from_bytes(message)
                    else:
                        serializer = TorchSerializer
                        request = TorchSerializer.from_bytes(message)
                    decode_ms = (time.perf_counter() - t_decode) * 1000.0

                    endpoint = request.get("endpoint", "inference")
                    if endpoint not in self._endpoints:
                        raise ValueError(f"Unknown endpoint: {endpoint}")
                    handler = self._endpoints[endpoint]
                    t_handler = time.perf_counter()
                    result = handler.handler(request.get("data", {})) if handler.requires_input else handler.handler()
                    handler_ms = (time.perf_counter() - t_handler) * 1000.0
                    if isinstance(result, dict):
                        timing = result.setdefault("_server_timing_ms", {})
                        timing["socket_decode_ms"] = decode_ms
                        timing["handler_ms"] = handler_ms
                        timing["request_bytes"] = len(message)
                    if serializer is PickleSerializer:
                        result = _to_pickle_safe(result)
                    t_send = time.perf_counter()
                    payload = serializer.to_bytes(result)
                    self._send_message(conn, payload)
                    send_ms = (time.perf_counter() - t_send) * 1000.0
                    if send_ms > 1000.0:
                        print(f"Warning: TCP response encode/send took {send_ms:.1f}ms")
                except Exception as exc:
                    print(f"Error in TCP server: {exc}")
                    import traceback

                    print(traceback.format_exc())
                    self._send_message(conn, b"ERROR")
                    break

    def run(self) -> None:
        print(f"Server is ready and listening on tcp://{self.host}:{self.port} (transport=tcp)")
        while self.running:
            try:
                conn, _ = self.server_socket.accept()
            except socket.timeout:
                continue
            _set_tcp_socket_options(conn)
            self._serve_connection(conn)
        self.server_socket.close()


class RobotTcpInferenceServer(BaseTcpInferenceServer):
    def __init__(self, model, host: str = "127.0.0.1", port: int = 5555):
        super().__init__(host, port)
        self.register_endpoint("inference", model.inference)


class BaseTcpInferenceClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 15000):
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _set_tcp_socket_options(self.socket)
        self.socket.settimeout(self.timeout_ms / 1000.0)
        self.socket.connect((self.host, self.port))

    @staticmethod
    def _recv_exact(conn: socket.socket, size: int) -> bytes:
        chunks = []
        received = 0
        while received < size:
            chunk = conn.recv(size - received)
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _recv_message(self) -> bytes:
        size = int.from_bytes(self._recv_exact(self.socket, 4), "big")
        return self._recv_exact(self.socket, size)

    def _send_message(self, payload: bytes) -> None:
        self.socket.sendall(len(payload).to_bytes(4, "big"))
        self.socket.sendall(payload)

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        self._send_message(TorchSerializer.to_bytes(request))
        message = self._recv_message()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return TorchSerializer.from_bytes(message)

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except Exception:
            return False

    def kill_server(self) -> None:
        self.call_endpoint("kill", requires_input=False)

    def close(self) -> None:
        self.socket.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RobotTcpInferenceClient(BaseTcpInferenceClient):
    def inference(self, observations: dict[str, Any]):
        return self.call_endpoint("inference", observations)


class RobotInferenceServer(RobotTcpInferenceServer):
    pass


class RobotInferenceClient(RobotTcpInferenceClient):
    pass
