"""Lexer para MSC-Lang 2.0."""

import hashlib
from typing import Any, List

from taec.core.monitoring import perf_monitor
from taec.core.cache import AdaptiveCache
from taec.mscl.tokens import MSCLTokenType, MSCLToken

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MSCLLexer:
    """Lexer con manejo de errores y caché."""

    def __init__(self, source: str, filename: str = "<mscl>"):
        self.source = source
        self.filename = filename
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[MSCLToken] = []
        self.indent_stack = [0]
        self.errors: List[str] = []
        self.warnings: List[str] = []

        self._token_cache = AdaptiveCache[List[MSCLToken]](max_size=100)

        self.keywords = {
            'synth': MSCLTokenType.SYNTH, 'node': MSCLTokenType.NODE,
            'flow': MSCLTokenType.FLOW, 'evolve': MSCLTokenType.EVOLVE,
            'merge': MSCLTokenType.MERGE, 'spawn': MSCLTokenType.SPAWN,
            'reflect': MSCLTokenType.REFLECT, 'quantum': MSCLTokenType.QUANTUM,
            'dream': MSCLTokenType.DREAM, 'function': MSCLTokenType.FUNCTION,
            'return': MSCLTokenType.RETURN, 'if': MSCLTokenType.IF,
            'else': MSCLTokenType.ELSE, 'while': MSCLTokenType.WHILE,
            'for': MSCLTokenType.FOR, 'in': MSCLTokenType.IN,
            'break': MSCLTokenType.BREAK, 'continue': MSCLTokenType.CONTINUE,
            'import': MSCLTokenType.IMPORT, 'as': MSCLTokenType.AS,
            'class': MSCLTokenType.CLASS, 'self': MSCLTokenType.SELF,
            'async': MSCLTokenType.ASYNC, 'await': MSCLTokenType.AWAIT,
            'try': MSCLTokenType.TRY, 'except': MSCLTokenType.EXCEPT,
            'finally': MSCLTokenType.FINALLY, 'with': MSCLTokenType.WITH,
            'lambda': MSCLTokenType.LAMBDA, 'yield': MSCLTokenType.YIELD,
            'and': MSCLTokenType.AND, 'or': MSCLTokenType.OR,
            'not': MSCLTokenType.NOT, 'true': MSCLTokenType.TRUE,
            'false': MSCLTokenType.FALSE, 'null': MSCLTokenType.NULL,
            'pattern': MSCLTokenType.PATTERN, 'match': MSCLTokenType.MATCH,
            'case': MSCLTokenType.CASE, 'category': MSCLTokenType.CATEGORY,
            'functor': MSCLTokenType.FUNCTOR, 'monad': MSCLTokenType.MONAD,
        }

        self.multi_char_ops = {
            '->': MSCLTokenType.CONNECT, '<->': MSCLTokenType.BICONNECT,
            '~>': MSCLTokenType.TRANSFORM, '=>': MSCLTokenType.EMERGE,
            '~~': MSCLTokenType.RESONATE, '>>': MSCLTokenType.COMPOSE,
            '|>': MSCLTokenType.PIPELINE, '**': MSCLTokenType.POWER,
            '==': MSCLTokenType.EQUALS, '!=': MSCLTokenType.NOT_EQUALS,
            '<=': MSCLTokenType.LESS_EQUAL, '>=': MSCLTokenType.GREATER_EQUAL,
            '+=': MSCLTokenType.PLUS_ASSIGN, '-=': MSCLTokenType.MINUS_ASSIGN,
        }

        self.single_char_ops = {
            '+': MSCLTokenType.PLUS, '-': MSCLTokenType.MINUS,
            '*': MSCLTokenType.MULTIPLY, '/': MSCLTokenType.DIVIDE,
            '%': MSCLTokenType.MODULO, '<': MSCLTokenType.LESS_THAN,
            '>': MSCLTokenType.GREATER_THAN, '=': MSCLTokenType.ASSIGN,
            '(': MSCLTokenType.LPAREN, ')': MSCLTokenType.RPAREN,
            '{': MSCLTokenType.LBRACE, '}': MSCLTokenType.RBRACE,
            '[': MSCLTokenType.LBRACKET, ']': MSCLTokenType.RBRACKET,
            ';': MSCLTokenType.SEMICOLON, ',': MSCLTokenType.COMMA,
            '.': MSCLTokenType.DOT, ':': MSCLTokenType.COLON,
        }

    def error(self, message: str):
        error_msg = f"{self.filename}:{self.line}:{self.column}: {message}"
        self.errors.append(error_msg)
        logger.error("Lexer error: %s", error_msg)

    def warning(self, message: str):
        warning_msg = f"{self.filename}:{self.line}:{self.column}: {message}"
        self.warnings.append(warning_msg)
        logger.warning("Lexer warning: %s", warning_msg)

    def tokenize(self) -> List[MSCLToken]:
        source_hash = hashlib.sha256(self.source.encode()).hexdigest()
        cached = self._token_cache.get(source_hash)
        if cached:
            return cached

        with perf_monitor.timer("lexer_tokenize"):
            lines = self.source.split('\n')

            for line_idx, line in enumerate(lines):
                self.line = line_idx + 1
                self.column = 1
                self.position = 0

                if line.strip():
                    indent_level = self._get_indent_level(line)
                    current_indent = self.indent_stack[-1]

                    if indent_level > current_indent:
                        self.indent_stack.append(indent_level)
                        self._add_token(MSCLTokenType.INDENT, indent_level)
                    elif indent_level < current_indent:
                        while self.indent_stack and self.indent_stack[-1] > indent_level:
                            self.indent_stack.pop()
                            self._add_token(MSCLTokenType.DEDENT, indent_level)
                        if self.indent_stack[-1] != indent_level:
                            self.error(f"Inconsistent indentation: expected {self.indent_stack[-1]}, got {indent_level}")

                self._tokenize_line(line.lstrip())

                if line_idx < len(lines) - 1:
                    self._add_token(MSCLTokenType.NEWLINE, '\n')

            while len(self.indent_stack) > 1:
                self.indent_stack.pop()
                self._add_token(MSCLTokenType.DEDENT, 0)

            self._add_token(MSCLTokenType.EOF, None)
            self._token_cache.put(source_hash, self.tokens)
            perf_monitor.increment_counter("tokens_generated", len(self.tokens))

            return self.tokens

    def _get_indent_level(self, line: str) -> int:
        indent = 0
        for char in line:
            if char == ' ':
                indent += 1
            elif char == '\t':
                indent += 4
                self.warning("Tab character used for indentation (converted to 4 spaces)")
            else:
                break
        return indent

    def _tokenize_line(self, line: str):
        self.position = 0
        while self.position < len(line):
            self._skip_whitespace(line)
            if self.position >= len(line):
                break
            if self._peek(line) == '#':
                self._skip_comment(line)
                continue
            if self._peek_ahead(line, 3) in ('"""', "'''"):
                self._read_multiline_string(line)
                continue
            if self._peek(line).isdigit() or (self._peek(line) == '.' and self._peek(line, 1).isdigit()):
                self._read_number(line)
            elif self._peek(line).isalpha() or self._peek(line) == '_':
                self._read_identifier(line)
            elif self._peek(line) in '"\'':
                self._read_string(line)
            else:
                found = False
                for op, token_type in sorted(self.multi_char_ops.items(), key=lambda x: -len(x[0])):
                    if line[self.position:].startswith(op):
                        self._add_token(token_type, op)
                        self._advance(len(op))
                        found = True
                        break
                if not found:
                    char = self._peek(line)
                    if char in self.single_char_ops:
                        self._add_token(self.single_char_ops[char], char)
                        self._advance()
                    else:
                        self.error(f"Unexpected character '{char}'")
                        self._advance()

    def _peek(self, line: str, offset: int = 0) -> str:
        pos = self.position + offset
        return line[pos] if pos < len(line) else '\0'

    def _peek_ahead(self, line: str, count: int) -> str:
        return line[self.position:self.position + count]

    def _advance(self, count: int = 1):
        self.position += count
        self.column += count

    def _skip_whitespace(self, line: str):
        while self.position < len(line) and line[self.position] in ' \t':
            self._advance()

    def _skip_comment(self, line: str):
        start_col = self.column
        self._advance()
        if self._peek(line) == '#':
            self._advance()
            comment_text = line[self.position:].strip()
            token = MSCLToken(MSCLTokenType.COMMENT, comment_text, self.line, start_col)
            token.metadata['is_doc'] = True
            self.tokens.append(token)
        while self.position < len(line):
            self._advance()

    def _read_multiline_string(self, line: str):
        start_col = self.column
        self._advance(3)
        self.warning("Multi-line strings not fully supported yet")
        value = line[self.position:]
        self._advance(len(value))
        token = MSCLToken(MSCLTokenType.STRING, value, self.line, start_col)
        token.metadata['multiline'] = True
        self.tokens.append(token)

    def _read_number(self, line: str):
        start_pos = self.position
        start_col = self.column

        if line[self.position:self.position + 2] in ('0b', '0o', '0x'):
            prefix = line[self.position:self.position + 2]
            self._advance(2)
            if prefix == '0b':
                while self.position < len(line) and line[self.position] in '01_':
                    self._advance()
            elif prefix == '0o':
                while self.position < len(line) and line[self.position] in '01234567_':
                    self._advance()
            else:
                while self.position < len(line) and line[self.position] in '0123456789abcdefABCDEF_':
                    self._advance()
        else:
            while self.position < len(line) and (line[self.position].isdigit() or line[self.position] == '_'):
                self._advance()
            if self.position < len(line) and line[self.position] == '.':
                self._advance()
                while self.position < len(line) and (line[self.position].isdigit() or line[self.position] == '_'):
                    self._advance()
            if self.position < len(line) and line[self.position] in 'eE':
                self._advance()
                if self.position < len(line) and line[self.position] in '+-':
                    self._advance()
                while self.position < len(line) and line[self.position].isdigit():
                    self._advance()
            if self.position < len(line) and line[self.position] in 'jJ':
                self._advance()

        value_str = line[start_pos:self.position].replace('_', '')
        try:
            if value_str.startswith('0b'):
                value = int(value_str, 2)
            elif value_str.startswith('0o'):
                value = int(value_str, 8)
            elif value_str.startswith('0x'):
                value = int(value_str, 16)
            elif 'j' in value_str.lower():
                value = complex(value_str)
            elif '.' in value_str or 'e' in value_str.lower():
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            self.error(f"Invalid number literal: {value_str}")
            value = 0

        token = MSCLToken(MSCLTokenType.NUMBER, value, self.line, start_col)
        token.end_column = self.column
        self.tokens.append(token)

    def _read_identifier(self, line: str):
        start_pos = self.position
        start_col = self.column
        if not (line[self.position].isalpha() or line[self.position] == '_'):
            self.error(f"Invalid identifier start: '{line[self.position]}'")
            return
        while self.position < len(line) and (line[self.position].isalnum() or line[self.position] == '_'):
            self._advance()
        value = line[start_pos:self.position]
        token_type = self.keywords.get(value, MSCLTokenType.IDENTIFIER)
        token = MSCLToken(token_type, value, self.line, start_col)
        token.end_column = self.column
        self.tokens.append(token)

    def _read_string(self, line: str):
        quote_char = line[self.position]
        start_col = self.column
        self._advance()
        is_fstring = self.position > 1 and line[self.position - 2] == 'f'
        value = ''
        while self.position < len(line) and line[self.position] != quote_char:
            if line[self.position] == '\\':
                self._advance()
                if self.position < len(line):
                    escape_char = line[self.position]
                    escape_map = {
                        'n': '\n', 't': '\t', 'r': '\r', '\\': '\\',
                        quote_char: quote_char, 'a': '\a', 'b': '\b',
                        'f': '\f', 'v': '\v', '0': '\0'
                    }
                    value += escape_map.get(escape_char, escape_char)
                    self._advance()
            else:
                value += line[self.position]
                self._advance()
        if self.position >= len(line):
            self.error("Unterminated string")
        else:
            self._advance()
        token = MSCLToken(MSCLTokenType.STRING, value, self.line, start_col)
        token.end_column = self.column
        if is_fstring:
            token.metadata['is_fstring'] = True
        self.tokens.append(token)

    def _add_token(self, token_type: MSCLTokenType, value: Any):
        token = MSCLToken(token_type, value, self.line, self.column, filename=self.filename)
        self.tokens.append(token)
