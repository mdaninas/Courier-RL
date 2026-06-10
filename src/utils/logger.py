from __future__ import annotations

import logging
import sys

_BASE = "courier_rl"
_CONFIGURED = False


def _configure_base(level: int) -> None:
    global _CONFIGURED
    base = logging.getLogger(_BASE)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        base.addHandler(handler)
        base.propagate = False
        _CONFIGURED = True
    base.setLevel(level)


def get_logger(name: str = _BASE, level: int = logging.INFO) -> logging.Logger:
    _configure_base(level)
    if name == _BASE or name.startswith(_BASE + "."):
        full = name
    else:
        full = f"{_BASE}.{name}"
    logger = logging.getLogger(full)
    logger.setLevel(level)
    return logger


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
