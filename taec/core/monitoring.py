"""Monitor de rendimiento para el sistema TAEC."""

import time
import threading
import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Any

import numpy as np

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor de rendimiento para el sistema TAEC."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.timers = {}
        self.counters = defaultdict(int)
        self.lock = threading.RLock()

    @contextmanager
    def timer(self, name: str):
        """Context manager para medir tiempo."""
        start = time.perf_counter()
        self.timers[name] = start
        try:
            yield
        finally:
            with self.lock:
                duration = time.perf_counter() - start
                self.metrics[f"{name}_duration"].append(duration)
                self.counters[f"{name}_count"] += 1

    def record_metric(self, name: str, value: float):
        """Registra una métrica."""
        with self.lock:
            self.metrics[name].append(value)

    def increment_counter(self, name: str, amount: int = 1):
        """Incrementa un contador."""
        with self.lock:
            self.counters[name] += amount

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas."""
        with self.lock:
            stats = {}

            for name, values in self.metrics.items():
                if values:
                    stats[name] = {
                        'count': len(values),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'p50': float(np.percentile(values, 50)),
                        'p95': float(np.percentile(values, 95)),
                        'p99': float(np.percentile(values, 99))
                    }

            stats['counters'] = dict(self.counters)

            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                stats['system'] = {
                    'cpu_percent': process.cpu_percent(),
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'threads': process.num_threads()
                }

            return stats


perf_monitor = PerformanceMonitor()
