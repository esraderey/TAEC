"""Historial de evolución con análisis."""

from collections import deque, Counter
from typing import Dict, Any, List, Optional

import numpy as np


class EvolutionHistory:
    """Historial de evolución con análisis."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.history: deque = deque(maxlen=max_size)
        self.index: Dict[str, Dict[str, Any]] = {}

    def add(self, entry: Dict[str, Any]):
        """Añade entrada al historial."""
        evolution_id = entry.get('id')
        if evolution_id:
            self.index[evolution_id] = entry
        self.history.append(entry)

    def get_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene entradas recientes."""
        return list(self.history)[-limit:]

    def get_by_id(self, evolution_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene entrada por ID."""
        return self.index.get(evolution_id)

    def analyze_history(self) -> Dict[str, Any]:
        """Analiza el historial completo."""
        if not self.history:
            return {'empty': True}

        total_entries = len(self.history)
        scores = [
            entry.get('success_metrics', {}).get('overall_score', 0)
            for entry in self.history
        ]
        strategies = [entry.get('strategy', 'unknown') for entry in self.history]
        strategy_counts = Counter(strategies)
        timestamps = [entry.get('timestamp', 0) for entry in self.history]
        if timestamps:
            time_span = max(timestamps) - min(timestamps)
            evolution_rate = total_entries / (time_span / 3600) if time_span > 0 else 0
        else:
            evolution_rate = 0

        return {
            'total_entries': total_entries,
            'average_score': float(np.mean(scores)) if scores else 0,
            'score_std': float(np.std(scores)) if scores else 0,
            'best_score': max(scores) if scores else 0,
            'worst_score': min(scores) if scores else 0,
            'strategy_distribution': dict(strategy_counts),
            'evolution_rate_per_hour': evolution_rate
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            'history': list(self.history)[-1000:],
            'stats': self.analyze_history()
        }

    def load_state(self, state: Dict[str, Any]):
        """Carga estado desde diccionario."""
        history_data = state.get('history', [])
        self.history.extend(history_data)
        for entry in self.history:
            if 'id' in entry:
                self.index[entry['id']] = entry

    def __len__(self) -> int:
        return len(self.history)

    def __bool__(self) -> bool:
        return len(self.history) > 0
