"""Permite ejecutar python -m taec."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def main():
    monolith_path = os.path.join(ROOT, "taec_v3_enhanced (1).py")
    if os.path.isfile(monolith_path):
        import runpy
        runpy.run_path(monolith_path, run_name="__main__")
    else:
        print("TAEC v3.0 - Ejecute run_taec.py o use: from taec import TAECAdvancedModule")

if __name__ == "__main__":
    main()
