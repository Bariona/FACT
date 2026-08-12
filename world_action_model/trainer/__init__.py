from .. import transformers

__all__ = [
    "CasualWATrainer",
]


def __getattr__(name):
    if name == "CasualWATrainer":
        from .wa_casual_trainer import CasualWATrainer

        return CasualWATrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
