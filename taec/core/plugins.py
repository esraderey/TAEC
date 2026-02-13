"""Sistema de plugins para extensibilidad TAEC."""

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Callable

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class TAECPlugin(ABC):
    """Clase base para plugins del sistema TAEC."""

    @abstractmethod
    def initialize(self, taec_module: 'TAECAdvancedModule'):
        """Inicializa el plugin."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Obtiene nombre del plugin."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Obtiene versión del plugin."""
        pass


class PluginManager:
    """Gestor de plugins para extensibilidad."""

    def __init__(self):
        self.plugins: Dict[str, TAECPlugin] = {}
        self.hooks: Dict[str, List[Callable]] = defaultdict(list)

    def register_plugin(self, plugin: TAECPlugin):
        """Registra un plugin."""
        name = plugin.get_name()
        if name in self.plugins:
            raise ValueError(f"Plugin {name} already registered")

        self.plugins[name] = plugin
        logger.info("Plugin registered: %s v%s", name, plugin.get_version())

    def register_hook(self, event: str, callback: Callable):
        """Registra un hook para un evento."""
        self.hooks[event].append(callback)

    async def trigger_hook(self, event: str, *args, **kwargs):
        """Dispara hooks para un evento."""
        for callback in self.hooks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error("Hook error in %s: %s", event, e)
