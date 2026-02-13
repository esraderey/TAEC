"""
Dependencias opcionales de TAEC: ML (torch), visualización (matplotlib, networkx) y sistema (psutil).

Todas las importaciones son opcionales; si faltan, los módulos son None y los flags *_AVAILABLE son False.
"""

# Dependencias opcionales: ML
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    optim = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = None  # type: ignore

# Dependencias opcionales: visualización
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import networkx as nx
    from matplotlib.animation import FuncAnimation
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = None  # type: ignore
    animation = None  # type: ignore
    nx = None  # type: ignore
    FuncAnimation = None  # type: ignore
    sns = None  # type: ignore

# Dependencias opcionales: monitor de sistema
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore
