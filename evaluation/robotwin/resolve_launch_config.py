#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import sys
from pathlib import Path

import yaml


VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    raise TypeError(f"Unsupported value type in launch config: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description="Resolve RobotWin launch config into shell exports.")
    parser.add_argument("--config", required=True, help="Path to the shared launch YAML.")
    parser.add_argument("--section", required=True, choices=["server", "client"], help="Config section to export.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Launch config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text()) or {}
    section = config.get(args.section, {})
    if not isinstance(section, dict):
        raise TypeError(f"Expected '{args.section}' section to be a mapping in {config_path}")

    for key, value in section.items():
        if not VALID_NAME.match(key):
            raise ValueError(f"Invalid shell variable name '{key}' in {config_path}")
        if key in os.environ:
            continue
        resolved = _normalize(value)
        sys.stdout.write(f"export {key}={shlex.quote(resolved)}\n")


if __name__ == "__main__":
    main()
