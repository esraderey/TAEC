#!/usr/bin/env python3
"""Extrae la clase TAECAdvancedModule del monolito y genera taec/advanced_module.py"""

import os

MONOLITH = os.path.join(os.path.dirname(__file__), "taec_v3_enhanced (1).py")
OUTPUT = os.path.join(os.path.dirname(__file__), "taec", "advanced_module.py")

HEADER = r'''"""
Módulo principal TAEC: clase TAECAdvancedModule.

Orquesta el sistema de auto-evolución cognitiva (evolución, memoria cuántica, MSC-Lang, plugins).
"""

import asyncio
import hashlib
import importlib.util
import inspect
import math
import os
import pickle
import random
import re
import time
import traceback
import zlib
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from taec.evolution import CodeEvolutionEngine, EvolutionHistory
from taec.core import perf_monitor, TAECPlugin, PluginManager
from taec.soporte import get_logger, VISUALIZATION_AVAILABLE, PSUTIL_AVAILABLE, plt, nx, psutil

logger = get_logger()

# Dependencias que siguen en el monolito (hasta extraerlas a sus módulos)
_legacy = None

def _get_legacy():
    global _legacy
    if _legacy is None:
        from taec.legacy_loader import _load_legacy
        _legacy = _load_legacy()
    return _legacy

def _QuantumVirtualMemory(*args, **kwargs):
    return _get_legacy().QuantumVirtualMemory(*args, **kwargs)

def _MSCLCompiler(*args, **kwargs):
    return _get_legacy().MSCLCompiler(*args, **kwargs)

TemplateManager = lambda: _get_legacy().TemplateManager()
CodeRepository = lambda *a, **k: _get_legacy().CodeRepository(*a, **k)
MetricsCollector = lambda: _get_legacy().MetricsCollector()
ImpactAnalyzer = lambda graph: _get_legacy().ImpactAnalyzer(graph)
StrategySelector = lambda: _get_legacy().StrategySelector()
EmergenceDetector = lambda graph: _get_legacy().EmergenceDetector(graph)
EmergencePattern = _get_legacy().EmergencePattern
QuantumCircuit = _get_legacy().QuantumCircuit
MSCLLexer = _get_legacy().MSCLLexer


@dataclass
class Event:
    """Evento para el bus del grafo."""
    type: str
    data: Any


'''

def main():
    with open(MONOLITH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Solo TAECAdvancedModule: líneas 2001-5827 (sin TemplateManager, CodeRepository, etc.)
    class_lines = lines[1999:5827]

    # Instanciar clases del legacy correctamente (no lambdas que devuelven clase)
    # En el __init__ se usa: QuantumVirtualMemory(...), TemplateManager(), CodeRepository(...), etc.
    # Por tanto necesitamos las clases, no instancias. Corregimos el header.
    body = "".join(class_lines)

    # Corregir bug: connections.extend(len(...)) -> connections.append(len(...))
    body = body.replace(
        "connections.extend(len(node.connections_out))",
        "connections.append(len(node.connections_out))"
    )
    # Corregir clave en _calculate_state_changes para que coincida con changes.get('state_changes', 0)
    body = body.replace(
        "'state_change': final['total_state'] - initial['total_state']",
        "'state_changes': final['total_state'] - initial['total_state']"
    )

    # Reemplazar referencias a clases legacy: usar las clases cargadas del legacy
    # En el cuerpo se usan: QuantumVirtualMemory, MSCLCompiler, TemplateManager(), CodeRepository(), MetricsCollector(), ImpactAnalyzer(self.graph), StrategySelector(), EmergenceDetector(self.graph), EmergencePattern, QuantumCircuit(), MSCLLexer
    # Si dejamos los nombres igual (QuantumVirtualMemory, etc.) tenemos que asignar en el header las clases desde legacy, no lambdas.
    header_fixed = r'''"""
Módulo principal TAEC: clase TAECAdvancedModule.

Orquesta el sistema de auto-evolución cognitiva (evolución, memoria cuántica, MSC-Lang, plugins).
"""

import asyncio
import hashlib
import importlib.util
import inspect
import math
import os
import pickle
import random
import re
import time
import traceback
import zlib
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from taec.evolution import CodeEvolutionEngine, EvolutionHistory
from taec.core import perf_monitor, TAECPlugin, PluginManager
from taec.soporte import get_logger, VISUALIZATION_AVAILABLE, PSUTIL_AVAILABLE, plt, nx, psutil
from taec.mscl import MSCLCompiler, MSCLLexer

logger = get_logger()

# Dependencias que siguen en el monolito (hasta extraerlas a sus módulos)
def _get_legacy():
    from taec.legacy_loader import _load_legacy
    return _load_legacy()

_legacy = _get_legacy()
QuantumVirtualMemory = _legacy.QuantumVirtualMemory
TemplateManager = _legacy.TemplateManager
CodeRepository = _legacy.CodeRepository
MetricsCollector = _legacy.MetricsCollector
ImpactAnalyzer = _legacy.ImpactAnalyzer
StrategySelector = _legacy.StrategySelector
EmergenceDetector = _legacy.EmergenceDetector
EmergencePattern = _legacy.EmergencePattern
QuantumCircuit = _legacy.QuantumCircuit


@dataclass
class Event:
    """Evento para el bus del grafo."""
    type: str
    data: Any


'''

    content = header_fixed + body

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Written {OUTPUT} ({len(content)} chars)")


if __name__ == "__main__":
    main()
