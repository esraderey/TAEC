#!/usr/bin/env python3
"""
Punto de entrada para TAEC v3.0.
Ejecuta el demo del módulo. Para usar el código refactorizado en carpetas,
importa desde el paquete taec.
"""

import sys
import os

# Asegurar que el directorio del proyecto está en el path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    """Ejecuta el demo del archivo monolítico o del paquete taec."""
    # Intentar ejecutar el archivo original si existe (comportamiento legacy)
    monolith_path = os.path.join(ROOT, "taec_v3_enhanced (1).py")
    if os.path.isfile(monolith_path):
        import runpy
        runpy.run_path(monolith_path, run_name="__main__")
    else:
        # Fallback: demo mínimo con el paquete refactorizado
        from taec import TAECAdvancedModule, TAECPlugin
        print("TAEC v3.0 - Usar paquete taec. Ejemplo mínimo:")
        print("  from taec import TAECAdvancedModule")
        print("  taec = TAECAdvancedModule(graph, config)")
        print("  compiled, errors, warnings = taec.compile_mscl_code(mscl_source)")


if __name__ == "__main__":
    main()
