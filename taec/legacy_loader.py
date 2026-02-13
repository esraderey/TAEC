"""
Carga el módulo TAEC desde el archivo monolítico original.
Permite usar el paquete taec manteniendo el comportamiento del script original.
"""

import os
import importlib.util

_TAEC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MONOLITH_PATH = os.path.join(_TAEC_ROOT, "taec_v3_enhanced (1).py")

_legacy_module = None


def _load_legacy():
    global _legacy_module
    if _legacy_module is not None:
        return _legacy_module
    if not os.path.isfile(_MONOLITH_PATH):
        raise ImportError(
            f"No se encontró el módulo TAEC monolítico en {_MONOLITH_PATH}. "
            "Coloque 'taec_v3_enhanced (1).py' en el directorio del proyecto."
        )
    spec = importlib.util.spec_from_file_location("taec_legacy", _MONOLITH_PATH)
    _legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_legacy_module)
    return _legacy_module


def __getattr__(name):
    if name in ("TAECAdvancedModule", "example_usage"):
        legacy = _load_legacy()
        return getattr(legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Exportaciones al importar "from taec.legacy_loader import ..."
def get_TAECAdvancedModule():
    return _load_legacy().TAECAdvancedModule


def get_example_usage():
    return _load_legacy().example_usage
