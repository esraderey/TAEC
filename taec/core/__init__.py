"""Núcleo TAEC: monitor de rendimiento, caché y sistema de plugins."""

from taec.core.monitoring import PerformanceMonitor, perf_monitor
from taec.core.cache import AdaptiveCache
from taec.core.plugins import TAECPlugin, PluginManager

__all__ = [
    "PerformanceMonitor",
    "perf_monitor",
    "AdaptiveCache",
    "TAECPlugin",
    "PluginManager",
]
