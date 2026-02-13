"""Compilador MSC-Lang: lexer -> parser -> codegen."""

from typing import Tuple, Optional, List

from taec.mscl.lexer import MSCLLexer
from taec.mscl.parser import MSCLParser
from taec.mscl.codegen import MSCLCodeGenerator
from taec.mscl.ast import Program


class MSCLCompiler:
    """Compilador MSC-Lang a Python."""

    def __init__(self, optimize: bool = True, debug: bool = False):
        self.optimize = optimize
        self.debug = debug

    def compile(self, source: str) -> Tuple[Optional[str], List[str], List[str]]:
        """
        Compila código MSC-Lang a Python.

        Returns:
            (código_python, errores, advertencias)
        """
        lexer = MSCLLexer(source)
        tokens = lexer.tokenize()

        if lexer.errors:
            return None, lexer.errors, lexer.warnings

        parser = MSCLParser(tokens)
        ast = parser.parse()

        if parser.errors:
            return None, lexer.errors + parser.errors, lexer.warnings + parser.warnings

        try:
            codegen = MSCLCodeGenerator(optimize=self.optimize, target="python")
            python_code = codegen.generate(ast)
            warnings = lexer.warnings + parser.warnings
            return python_code, [], warnings
        except Exception as e:
            return None, [str(e)], lexer.warnings + parser.warnings
