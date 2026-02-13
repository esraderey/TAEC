"""Dataset y predictor de fitness para evolución (opcional, requiere PyTorch)."""

from typing import List

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    Dataset = object


if TORCH_AVAILABLE:

    class FitnessDataset(Dataset):
        """Dataset para entrenar predictor de fitness."""

        def __init__(self, max_size: int = 10000):
            self.features: List[List[float]] = []
            self.targets: List[float] = []
            self.max_size = max_size

        def add_sample(self, features: List[float], fitness: float):
            """Añade muestra al dataset."""
            self.features.append(features)
            self.targets.append(fitness)
            if len(self.features) > self.max_size:
                self.features = self.features[-self.max_size:]
                self.targets = self.targets[-self.max_size:]

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return (
                torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32)
            )

    class FitnessPredictor(nn.Module):
        """Red neuronal para predecir fitness de código."""

        def __init__(self, input_size: int = 30, hidden_sizes: List[int] = None):
            super().__init__()
            if hidden_sizes is None:
                hidden_sizes = [64, 32, 16]
            layers = []
            prev_size = input_size
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.2)
                ])
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

else:
    FitnessDataset = None  # type: ignore
    FitnessPredictor = None  # type: ignore
