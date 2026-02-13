"""
Módulo de soporte TAEC: dependencias opcionales y logging.

- deps: TORCH_AVAILABLE, VISUALIZATION_AVAILABLE, PSUTIL_AVAILABLE y módulos opcionales (torch, plt, nx, psutil, ...)
- logging: get_logger()
"""

from taec.soporte.deps import (
    TORCH_AVAILABLE,
    VISUALIZATION_AVAILABLE,
    PSUTIL_AVAILABLE,
    torch,
    nn,
    F,
    optim,
    DataLoader,
    Dataset,
    plt,
    animation,
    nx,
    FuncAnimation,
    sns,
    psutil,
)
from taec.soporte.logging import get_logger

__all__ = [
    "TORCH_AVAILABLE",
    "VISUALIZATION_AVAILABLE",
    "PSUTIL_AVAILABLE",
    "get_logger",
    "torch",
    "nn",
    "F",
    "optim",
    "DataLoader",
    "Dataset",
    "plt",
    "animation",
    "nx",
    "FuncAnimation",
    "sns",
    "psutil",
]
