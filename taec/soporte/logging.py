"""
Logging unificado para TAEC: structlog si está disponible, si no logging estándar.
"""

import logging


def get_logger():
    """Logger unificado: structlog si está disponible, si no logging estándar."""
    try:
        import structlog
        return structlog.get_logger()
    except ImportError:
        return logging.getLogger("taec")
