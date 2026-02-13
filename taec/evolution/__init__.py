"""
Motor de evolución de código TAEC.

- CodeEvolutionEngine: motor con estrategias standard, island, coevolution, novelty_search
- EvolutionHistory: historial de ciclos de evolución con análisis
- Operadores genéticos: AdaptiveMutation, SemanticCrossover
"""

from taec.evolution.engine import CodeEvolutionEngine
from taec.evolution.history import EvolutionHistory
from taec.evolution.operators import GeneticOperator, AdaptiveMutation, SemanticCrossover

try:
    from taec.evolution.fitness_ml import (
        TORCH_AVAILABLE,
        FitnessDataset,
        FitnessPredictor,
    )
except ImportError:
    TORCH_AVAILABLE = False
    FitnessDataset = None
    FitnessPredictor = None

__all__ = [
    "CodeEvolutionEngine",
    "EvolutionHistory",
    "GeneticOperator",
    "AdaptiveMutation",
    "SemanticCrossover",
    "TORCH_AVAILABLE",
    "FitnessDataset",
    "FitnessPredictor",
]
