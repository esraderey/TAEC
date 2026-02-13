"""Parser para MSC-Lang 2.0."""

from typing import List

from taec.core.monitoring import perf_monitor
from taec.core.cache import AdaptiveCache
from taec.mscl.tokens import MSCLTokenType, MSCLToken
from taec.mscl.ast import (
    Program,
    PatternMatch,
    MatchCase,
    CategoryDef,
    MSCLASTNode,
    Identifier,
    Call,
    Literal,
)

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MSCLParser:
    """Parser recursivo descendente para MSC-Lang 2.0."""

    def __init__(self, tokens: List[MSCLToken]):
        self.tokens = tokens
        self.position = 0
        self.current_token = self.tokens[0] if tokens else None
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self._parse_cache = AdaptiveCache[MSCLASTNode](max_size=500)
        self._recovery_stack = []

    def advance(self):
        self.position += 1
        self.current_token = self.tokens[self.position] if self.position < len(self.tokens) else None

    def match(self, *types: MSCLTokenType) -> bool:
        return self.current_token is not None and self.current_token.type in types

    def consume(self, token_type: MSCLTokenType):
        if self.match(token_type):
            t = self.current_token
            self.advance()
            return t
        msg = f"Expected {token_type}"
        if self.current_token:
            msg = f"Parse error at line {self.current_token.line}, column {self.current_token.column}: {msg}"
        self.errors.append(msg)
        logger.error(msg)
        self._try_recovery()
        return None

    def skip_newlines(self):
        while self.current_token and self.current_token.type in (
            MSCLTokenType.NEWLINE, MSCLTokenType.INDENT, MSCLTokenType.DEDENT
        ):
            self.advance()

    def error(self, message: str, recoverable: bool = True):
        if self.current_token:
            error_msg = f"Parse error at line {self.current_token.line}, column {self.current_token.column}: {message}"
        else:
            error_msg = f"Parse error: {message}"
        self.errors.append(error_msg)
        logger.error(error_msg)
        if recoverable:
            self._try_recovery()

    def warning(self, message: str):
        if self.current_token:
            warning_msg = f"Parse warning at line {self.current_token.line}, column {self.current_token.column}: {message}"
        else:
            warning_msg = f"Parse warning: {message}"
        self.warnings.append(warning_msg)
        logger.warning(warning_msg)

    def _try_recovery(self):
        sync_tokens = {
            MSCLTokenType.SEMICOLON, MSCLTokenType.NEWLINE, MSCLTokenType.RBRACE,
            MSCLTokenType.FUNCTION, MSCLTokenType.CLASS, MSCLTokenType.IF,
            MSCLTokenType.WHILE, MSCLTokenType.FOR, MSCLTokenType.EOF
        }
        while self.current_token and self.current_token.type not in sync_tokens:
            self.advance()
        if self.match(MSCLTokenType.SEMICOLON, MSCLTokenType.NEWLINE):
            self.advance()

    def parse(self) -> Program:
        statements = []
        with perf_monitor.timer("parser_parse"):
            while not self.match(MSCLTokenType.EOF):
                self.skip_newlines()
                if self.match(MSCLTokenType.EOF):
                    break
                try:
                    stmt = self.parse_statement()
                    if stmt is not None:
                        statements.append(stmt)
                except SyntaxError as e:
                    logger.error("Fatal parse error: %s", e)
                    break
                self.skip_newlines()
        perf_monitor.increment_counter("ast_nodes_created", len(statements))
        return Program(statements)

    def parse_statement(self):
        if self.match(MSCLTokenType.EOF):
            return None
        if self.match(MSCLTokenType.MATCH):
            return self.parse_pattern_match()
        if self.match(MSCLTokenType.CATEGORY):
            return self.parse_category()
        self._try_recovery()
        return None

    def parse_expression(self):
        return self.parse_primary()

    def parse_primary(self):
        if self.match(MSCLTokenType.NUMBER):
            v = self.current_token.value
            self.advance()
            return Literal(v)
        if self.match(MSCLTokenType.IDENTIFIER):
            name = self.current_token.value
            self.advance()
            return Identifier(name)
        if self.match(MSCLTokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.consume(MSCLTokenType.RPAREN)
            return expr
        if self.current_token:
            self.advance()
        return Literal(0)

    def parse_block(self) -> List[MSCLASTNode]:
        block = []
        while self.current_token and self.current_token.type not in (MSCLTokenType.RBRACE, MSCLTokenType.EOF):
            stmt = self.parse_statement()
            if stmt is not None:
                block.append(stmt)
            self.skip_newlines()
        return block

    def parse_pattern_match(self) -> PatternMatch:
        self.consume(MSCLTokenType.MATCH)
        subject = self.parse_expression()
        self.consume(MSCLTokenType.LBRACE)
        cases = []
        while self.current_token and not self.match(MSCLTokenType.RBRACE):
            self.skip_newlines()
            if self.match(MSCLTokenType.CASE):
                cases.append(self.parse_match_case())
            self.skip_newlines()
        self.consume(MSCLTokenType.RBRACE)
        if not cases:
            self.warning("Empty match expression")
        return PatternMatch(subject, cases)

    def parse_match_case(self) -> MatchCase:
        self.consume(MSCLTokenType.CASE)
        pattern = self.parse_pattern()
        guard = None
        if self.match(MSCLTokenType.IF):
            self.advance()
            guard = self.parse_expression()
        self.consume(MSCLTokenType.ARROW)
        if self.match(MSCLTokenType.LBRACE):
            self.advance()
            body = self.parse_block()
            self.consume(MSCLTokenType.RBRACE)
        else:
            body = [self.parse_expression()]
        return MatchCase(pattern, guard, body)

    def parse_pattern(self) -> MSCLASTNode:
        if self.match(MSCLTokenType.IDENTIFIER):
            name = self.current_token.value
            self.advance()
            if self.match(MSCLTokenType.LPAREN):
                self.advance()
                args = []
                while not self.match(MSCLTokenType.RPAREN):
                    args.append(self.parse_pattern())
                    if self.match(MSCLTokenType.COMMA):
                        self.advance()
                self.consume(MSCLTokenType.RPAREN)
                return Call(Identifier(name), args)
            return Identifier(name)
        return self.parse_primary()

    def parse_category(self) -> CategoryDef:
        self.consume(MSCLTokenType.CATEGORY)
        name = self.consume(MSCLTokenType.IDENTIFIER)
        name = name.value if name else "Unknown"
        self.consume(MSCLTokenType.LBRACE)
        objects = []
        morphisms = []
        while self.current_token and not self.match(MSCLTokenType.RBRACE):
            self.skip_newlines()
            if self.match(MSCLTokenType.IDENTIFIER):
                first = self.current_token.value
                self.advance()
                if self.match(MSCLTokenType.COLON):
                    self.advance()
                    src = self.consume(MSCLTokenType.IDENTIFIER)
                    self.consume(MSCLTokenType.CONNECT)
                    tgt = self.consume(MSCLTokenType.IDENTIFIER)
                    if src and tgt:
                        morphisms.append((first, src.value, tgt.value))
                else:
                    objects.append(first)
            if self.match(MSCLTokenType.SEMICOLON):
                self.advance()
            self.skip_newlines()
        self.consume(MSCLTokenType.RBRACE)
        return CategoryDef(name, objects, morphisms)
