"""
TAEC Advanced Module v3.0 - Sistema de Auto-Evolución Cognitiva Mejorado

Paquete refactorizado con arquitectura modular.
- taec.core: monitor de rendimiento, caché, plugins
- taec.mscl: lenguaje MSC-Lang 2.0 (tokens, lexer, parser, AST, compilador)
- taec.memory: memoria cuántica virtual
"""

from taec.core.plugins import TAECPlugin, PluginManager

__version__ = "3.0"

# TAECAdvancedModule desde el módulo dedicado (que a su vez usa el monolito legacy)
try:
    from taec.advanced_module import TAECAdvancedModule
except ImportError:
    TAECAdvancedModule = None  # type: ignore

try:
    from taec.legacy_loader import _load_legacy
    example_usage = getattr(_load_legacy(), "example_usage", None)
except ImportError:
    example_usage = None

__all__ = ["TAECAdvancedModule", "TAECPlugin", "PluginManager", "example_usage", "__version__"]
