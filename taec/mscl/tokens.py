"""Tipos de token y token para MSC-Lang 2.0."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict


class MSCLTokenType(Enum):
    """Tipos de token extendidos para MSC-Lang 2.0."""
    SYNTH = "synth"
    NODE = "node"
    FLOW = "flow"
    EVOLVE = "evolve"
    MERGE = "merge"
    SPAWN = "spawn"
    REFLECT = "reflect"
    QUANTUM = "quantum"
    DREAM = "dream"

    FUNCTION = "function"
    RETURN = "return"
    IF = "if"
    ELSE = "else"
    WHILE = "while"
    FOR = "for"
    IN = "in"
    BREAK = "break"
    CONTINUE = "continue"
    IMPORT = "import"
    AS = "as"
    CLASS = "class"
    SELF = "self"
    ASYNC = "async"
    AWAIT = "await"
    TRY = "try"
    EXCEPT = "except"
    FINALLY = "finally"
    WITH = "with"
    LAMBDA = "lambda"
    YIELD = "yield"

    PATTERN = "pattern"
    MATCH = "match"
    CASE = "case"
    CATEGORY = "category"
    FUNCTOR = "functor"
    MONAD = "monad"

    CONNECT = "->"
    BICONNECT = "<->"
    TRANSFORM = "~>"
    EMERGE = "=>"
    RESONATE = "~~"
    COMPOSE = ">>"
    PIPELINE = "|>"
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    POWER = "**"
    EQUALS = "=="
    NOT_EQUALS = "!="
    LESS_THAN = "<"
    GREATER_THAN = ">"
    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    AND = "and"
    OR = "or"
    NOT = "not"
    ASSIGN = "="
    PLUS_ASSIGN = "+="
    MINUS_ASSIGN = "-="

    NUMBER = "NUMBER"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"
    TRUE = "true"
    FALSE = "false"
    NULL = "null"

    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    SEMICOLON = ";"
    COMMA = ","
    DOT = "."
    COLON = ":"
    ARROW = "=>"

    EOF = "EOF"
    NEWLINE = "NEWLINE"
    INDENT = "INDENT"
    DEDENT = "DEDENT"
    COMMENT = "COMMENT"


@dataclass
class MSCLToken:
    """Token con posición y metadata."""
    type: MSCLTokenType
    value: Any
    line: int
    column: int
    end_line: int = 0
    end_column: int = 0
    filename: str = "<mscl>"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.end_line == 0:
            self.end_line = self.line
        if self.end_column == 0:
            self.end_column = self.column + len(str(self.value))
