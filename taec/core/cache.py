"""Sistema de caché adaptativo."""

import time
import threading
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any, TypeVar, Generic

T = TypeVar('T')


class AdaptiveCache(Generic[T]):
    """Sistema de caché adaptativo con múltiples estrategias."""

    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, Tuple[T, float, int]] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[T]:
        """Obtiene valor del caché."""
        with self.lock:
            if key in self.cache:
                value, timestamp, hits = self.cache[key]

                if self.ttl and time.time() - timestamp > self.ttl:
                    del self.cache[key]
                    self.miss_count += 1
                    return None

                self.cache[key] = (value, timestamp, hits + 1)
                self.cache.move_to_end(key)
                self.hit_count += 1
                return value

            self.miss_count += 1
            return None

    def put(self, key: str, value: T):
        """Almacena valor en caché."""
        with self.lock:
            while len(self.cache) >= self.max_size:
                self._evict()

            self.cache[key] = (value, time.time(), 0)
            self.cache.move_to_end(key)

    def _evict(self):
        """Estrategia de evicción adaptativa."""
        candidates = []

        for key, (value, timestamp, hits) in self.cache.items():
            age = time.time() - timestamp
            score = hits / (age + 1)
            candidates.append((score, key))

        candidates.sort()
        _, evict_key = candidates[0]
        del self.cache[evict_key]
        self.eviction_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'eviction_count': self.eviction_count
            }
