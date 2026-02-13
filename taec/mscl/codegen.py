"""Generador de código Python desde AST MSC-Lang."""

import hashlib
from typing import Set, Callable, Any

from taec.core.monitoring import perf_monitor
from taec.core.cache import AdaptiveCache
from taec.mscl.ast import (
    Program,
    FunctionDef,
    ClassDef,
    MSCLASTNode,
    MSCLTokenType,
)

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class CodeOptimizer:
    """Optimizador de código para el AST."""

    def __init__(self):
        self.optimizations = {}

    def optimize(self, ast: MSCLASTNode, level: int = 1) -> MSCLASTNode:
        return ast


class MSCLCodeGenerator:
    """Generador de código con soporte JIT."""

    def __init__(self, optimize: bool = True, target: str = "python"):
        self.optimize = optimize
        self.target = target
        self.output: list = []
        self.indent_level = 0
        self.temp_counter = 0
        self.imports: Set[str] = set()
        self.optimizer = CodeOptimizer()
        self._jit_cache = AdaptiveCache[Callable](max_size=100)

    def generate(self, ast: Program) -> str:
        with perf_monitor.timer("code_generation"):
            if self.optimize:
                ast = self.optimizer.optimize(ast, level=2)
            if self.target == "python":
                return self._generate_python(ast)
            raise ValueError(f"Unknown target: {self.target}")

    def _generate_python(self, ast: Program) -> str:
        self.output = []
        self.imports = set()
        ast.accept(self)
        import_lines = sorted(f"import {imp}" for imp in self.imports)
        if import_lines:
            import_lines.append("")
        generated_code = "\n".join(import_lines + self.output)
        if self.optimize:
            self._try_jit_compile(generated_code)
        return generated_code

    def _try_jit_compile(self, code: str):
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        if self._jit_cache.get(code_hash):
            return
        try:
            compiled = compile(code, '<mscl-jit>', 'exec')
            namespace = {}
            exec(compiled, namespace)
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    self._jit_cache.put(f"{code_hash}:{name}", obj)
            logger.info("JIT compiled %s objects", len([k for k in namespace if callable(namespace.get(k))]))
        except Exception as e:
            logger.warning("JIT compilation failed: %s", e)

    def visit_Program(self, node: Program) -> Any:
        for stmt in node.statements:
            stmt.accept(self)
        return None

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        async_str = "async " if node.is_async else ""
        params = ", ".join(node.params)
        self.output.append(f"{async_str}def {node.name}({params}):")
        self.indent_level += 1
        for s in node.body:
            self.output.append("    " * self.indent_level + "pass  # MSC-Lang")
        self.indent_level -= 1
        return None

    def visit_ClassDef(self, node: ClassDef) -> Any:
        bases = ", ".join(node.bases) if node.bases else ""
        self.output.append(f"class {node.name}({bases}):")
        self.indent_level += 1
        for s in node.body:
            self.output.append("    " * self.indent_level + "pass  # MSC-Lang")
        self.indent_level -= 1
        return None

    def visit_PatternMatch(self, node) -> Any:
        self.output.append("# pattern match -> Python match/case")
        return None

    def visit_MatchCase(self, node) -> Any:
        return None

    def visit_CategoryDef(self, node) -> Any:
        self.output.append(f"# category {node.name}")
        return None

    def visit_FunctorDef(self, node) -> Any:
        return None

    def visit_Identifier(self, node) -> Any:
        return None

    def visit_Call(self, node) -> Any:
        return None

    def visit_Literal(self, node) -> Any:
        return None

    def visit_BinaryOp(self, node) -> Any:
        return None
