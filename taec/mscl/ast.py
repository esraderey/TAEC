"""Nodos AST para MSC-Lang 2.0."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Type, TypeVar, Iterator

from taec.mscl.tokens import MSCLTokenType

TNode = TypeVar('TNode', bound='MSCLASTNode')


class ASTVisitor(ABC):
    """Visitor base para recorrido del AST."""

    def visit(self, node: 'MSCLASTNode'):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: 'MSCLASTNode'):
        for child in node.get_children():
            self.visit(child)


class MSCLASTNode(ABC):
    """Nodo base del AST."""

    def __init__(self, line: int = 0, column: int = 0):
        self.line = line
        self.column = column
        self.parent: Optional['MSCLASTNode'] = None
        self.metadata: Dict[str, Any] = {}
        self._hash_cache: Optional[int] = None

    @abstractmethod
    def accept(self, visitor):
        pass

    def add_child(self, child: 'MSCLASTNode'):
        child.parent = self

    def get_children(self) -> List['MSCLASTNode']:
        return []

    def walk(self) -> Iterator['MSCLASTNode']:
        yield self
        for child in self.get_children():
            yield from child.walk()

    def find_all(self, node_type: Type[TNode]) -> List[TNode]:
        return [node for node in self.walk() if isinstance(node, node_type)]


@dataclass
class Program(MSCLASTNode):
    """Nodo raíz del programa."""
    statements: List[MSCLASTNode]

    def accept(self, visitor):
        return visitor.visit_Program(self)

    def get_children(self):
        return self.statements


@dataclass
class FunctionDef(MSCLASTNode):
    name: str
    params: List[str]
    body: List[MSCLASTNode]
    is_async: bool = False
    decorators: List[MSCLASTNode] = field(default_factory=list)
    return_type: Optional[str] = None
    param_types: Dict[str, str] = field(default_factory=dict)
    docstring: Optional[str] = None

    def accept(self, visitor):
        return visitor.visit_FunctionDef(self)

    def get_children(self):
        return self.decorators + self.body


@dataclass
class ClassDef(MSCLASTNode):
    name: str
    bases: List[str]
    body: List[MSCLASTNode]
    metaclass: Optional[str] = None
    decorators: List[MSCLASTNode] = field(default_factory=list)
    docstring: Optional[str] = None

    def accept(self, visitor):
        return visitor.visit_ClassDef(self)

    def get_children(self):
        return self.decorators + self.body


@dataclass
class PatternMatch(MSCLASTNode):
    subject: MSCLASTNode
    cases: List['MatchCase']

    def accept(self, visitor):
        return visitor.visit_PatternMatch(self)

    def get_children(self):
        return [self.subject] + self.cases


@dataclass
class MatchCase(MSCLASTNode):
    pattern: MSCLASTNode
    guard: Optional[MSCLASTNode]
    body: List[MSCLASTNode]

    def accept(self, visitor):
        return visitor.visit_MatchCase(self)

    def get_children(self):
        children = [self.pattern] + self.body
        if self.guard:
            children.append(self.guard)
        return children


@dataclass
class CategoryDef(MSCLASTNode):
    name: str
    objects: List[str]
    morphisms: List[Tuple[str, str, str]]

    def accept(self, visitor):
        return visitor.visit_CategoryDef(self)

    def get_children(self):
        return []


@dataclass
class FunctorDef(MSCLASTNode):
    name: str
    source_category: str
    target_category: str
    object_map: Dict[str, str]
    morphism_map: Dict[str, str]

    def accept(self, visitor):
        return visitor.visit_FunctorDef(self)

    def get_children(self):
        return []


# Nodos usados por TypeInference/SemanticAnalyzer y parser
@dataclass
class Identifier(MSCLASTNode):
    name: str

    def accept(self, visitor):
        return visitor.visit_Identifier(self)

    def get_children(self):
        return []


@dataclass
class Call(MSCLASTNode):
    func: MSCLASTNode
    args: List[MSCLASTNode]

    def accept(self, visitor):
        return visitor.visit_Call(self)

    def get_children(self):
        return [self.func] + self.args


@dataclass
class Literal(MSCLASTNode):
    value: Any

    def accept(self, visitor):
        return visitor.visit_Literal(self)

    def get_children(self):
        return []


@dataclass
class BinaryOp(MSCLASTNode):
    left: MSCLASTNode
    op: MSCLTokenType
    right: MSCLASTNode

    def accept(self, visitor):
        return visitor.visit_BinaryOp(self)

    def get_children(self):
        return [self.left, self.right]
