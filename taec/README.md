# Paquete TAEC v3.0 (refactorizado)

Estructura del proyecto refactorizado a partir de `taec_v3_enhanced (1).py`.

## Estructura de carpetas

```
taec/
├── __init__.py          # API pública: TAECAdvancedModule, TAECPlugin, PluginManager
├── advanced_module.py   # Módulo de la clase TAECAdvancedModule (carga desde legacy)
├── __main__.py          # python -m taec
├── legacy_loader.py     # Carga el módulo monolítico original
├── core/                # Núcleo del sistema
│   ├── monitoring.py    # PerformanceMonitor, perf_monitor
│   ├── cache.py        # AdaptiveCache
│   └── plugins.py      # TAECPlugin, PluginManager
├── soporte/             # Dependencias opcionales y logging
│   ├── __init__.py     # Re-exporta todo (API pública)
│   ├── deps.py         # TORCH_AVAILABLE, VISUALIZATION_AVAILABLE, PSUTIL_AVAILABLE, torch, plt, nx, psutil, ...
│   └── logging.py      # get_logger()
├── mscl/                # Lenguaje MSC-Lang 2.0
│   ├── tokens.py       # MSCLTokenType, MSCLToken
│   ├── lexer.py        # MSCLLexer
│   ├── ast.py          # Nodos AST (Program, FunctionDef, ...)
│   ├── parser.py       # MSCLParser
│   ├── codegen.py      # CodeOptimizer, MSCLCodeGenerator
│   └── compiler.py     # MSCLCompiler
├── memory/              # Memoria cuántica virtual
│   ├── quantum_memory.py
│   └── ...
└── evolution/           # Motor de evolución de código
    ├── engine.py        # CodeEvolutionEngine (estrategias: standard, island, coevolution, novelty_search)
    ├── history.py       # EvolutionHistory
    ├── operators.py    # GeneticOperator, AdaptiveMutation, SemanticCrossover
    ├── fitness_ml.py    # FitnessDataset, FitnessPredictor (opcional, PyTorch)
    └── __init__.py
```

## Uso

- **Ejecutar demo (archivo original):**  
  `python run_taec.py` o `python "taec_v3_enhanced (1).py"`

- **Importar desde el paquete:**  
  `from taec import TAECAdvancedModule, TAECPlugin`  
  (carga el módulo monolítico si existe)

- **Usar solo partes refactorizadas:**  
  `from taec.core import perf_monitor, AdaptiveCache`  
  `from taec.mscl import MSCLCompiler`  
  `from taec.memory import QuantumVirtualMemory`  
  `from taec.evolution import CodeEvolutionEngine, EvolutionHistory`

## Notas

- El archivo `taec_v3_enhanced (1).py` sigue siendo el módulo principal que contiene `TAECAdvancedModule` y el demo; **importa** desde `taec.evolution`, `taec.core` y `taec.soporte`.
- Los submódulos `taec.core`, `taec.soporte`, `taec.mscl`, `taec.memory` y `taec.evolution` son código extraído y reutilizables de forma independiente.

## Módulos pendientes de refactorizar (monolito)

El monolito `taec_v3_enhanced (1).py` aún contiene código que se puede extraer a `taec`:

| Módulo / área | Clases / contenido | Destino sugerido |
|---------------|-------------------|------------------|
| **MSC-Lang (completo)** | MSCLTokenType, MSCLToken, MSCLLexer, AST (MSCLASTNode, Program, FunctionDef, ...), MSCLParser, TypeInference, SemanticAnalyzer, DataFlowAnalyzer, SymbolTable, CodeOptimizer, MSCLCodeGenerator | `taec.mscl` (ampliar; el parser/codegen del monolito tiene más características que el actual) |
| **Memoria cuántica** | QuantumErrorCorrection, QuantumCircuit, QuantumState, QuantumMemoryCell, MemoryLayer, BloomFilter, QuantumVirtualMemory, TensorNetwork | Parcialmente en `taec.memory`; revisar si el monolito tiene una versión extendida |
| **Evolución adaptativa** | AdaptiveEvolutionEngine, EmergenceDetector (clases grandes con sintaxis tipo Kotlin) | `taec.evolution` (ampliar o nuevo subpaquete) |
| **Servicios / utilidades** | TemplateManager, CodeRepository, MetricsCollector, ImpactAnalyzer, StrategySelector, EmergencePattern, EvolutionOptimizer | Nuevo módulo p.ej. `taec.services` o `taec.analysis` |
| **TAECAdvancedModule** | Clase principal que orquesta todo | Expuesto en `taec.advanced_module`; implementación aún en el monolito (vía `taec.legacy_loader`) |

Orden recomendado para seguir refactorizando: 1) Unificar MSC en `taec.mscl` (parser/codegen extendido); 2) Extraer servicios (TemplateManager, MetricsCollector, etc.); 3) Integrar evolución adaptativa y emergence en `taec.evolution`.
