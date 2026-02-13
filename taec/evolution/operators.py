"""Operadores genéticos para el motor de evolución de código."""

import ast
import re
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Any, List, Tuple

import numpy as np


class GeneticOperator(ABC):
    """Operador genético base."""

    @abstractmethod
    def apply(self, individual: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica el operador al individuo."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Obtiene nombre del operador."""
        pass


class AdaptiveMutation(GeneticOperator):
    """Mutación adaptativa que ajusta su tasa según el fitness."""

    def __init__(self, base_rate: float = 0.1):
        self.base_rate = base_rate
        self.success_history = deque(maxlen=100)

    def apply(self, individual: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica mutación adaptativa."""
        if self.success_history:
            success_rate = sum(self.success_history) / len(self.success_history)
            mutation_rate = self.base_rate * (2 - success_rate)
        else:
            mutation_rate = self.base_rate

        mutated = individual.copy()
        if 'mutations' not in mutated:
            mutated['mutations'] = list(individual.get('mutations', []))

        if random.random() < mutation_rate:
            mutation_type = random.choice(['constant', 'operator', 'structure', 'semantic'])

            if mutation_type == 'constant':
                mutated['code'] = self._mutate_constants(mutated['code'])
            elif mutation_type == 'operator':
                mutated['code'] = self._mutate_operators(mutated['code'])
            elif mutation_type == 'structure':
                mutated['code'] = self._mutate_structure(mutated['code'])
            else:
                mutated['code'] = self._mutate_semantic(mutated['code'], context)

            mutated['mutations'].append(f"{self.get_name()}:{mutation_type}")

        return mutated

    def _mutate_constants(self, code: str) -> str:
        """Muta constantes con distribución adaptativa."""
        def replace_number(match):
            value = float(match.group(0))
            if abs(value) < 1:
                std_dev = 0.1
            elif abs(value) < 10:
                std_dev = 0.5
            else:
                std_dev = abs(value) * 0.1
            mutation = np.random.normal(0, std_dev)
            new_value = value + mutation
            if '.' not in match.group(0):
                new_value = int(new_value)
            return str(new_value)

        return re.sub(r'\b\d+\.?\d*\b', replace_number, code)

    def _mutate_operators(self, code: str) -> str:
        """Muta operadores preservando semántica cuando es posible."""
        operator_groups = {
            'comparison': {
                '==': ['!=', '>=', '<='],
                '!=': ['=='],
                '<': ['<=', '!='],
                '>': ['>=', '!='],
                '<=': ['<', '=='],
                '>=': ['>', '==']
            },
            'arithmetic': {
                '+': ['-', '*'],
                '-': ['+'],
                '*': ['/', '+'],
                '/': ['*']
            },
            'logical': {
                'and': ['or'],
                'or': ['and']
            }
        }

        mutated = code
        group_name = random.choice(list(operator_groups.keys()))
        group = operator_groups[group_name]

        for op, replacements in group.items():
            if op in code and random.random() < 0.3:
                replacement = random.choice(replacements)
                occurrences = [m.start() for m in re.finditer(re.escape(op), code)]
                if occurrences:
                    pos = random.choice(occurrences)
                    mutated = code[:pos] + replacement + code[pos + len(op):]
                    break

        return mutated

    def _mutate_structure(self, code: str) -> str:
        """Mutación estructural inteligente."""
        lines = code.split('\n')
        mutation_type = random.choices(
            ['swap', 'duplicate', 'delete', 'indent', 'extract_function', 'inline'],
            weights=[0.3, 0.2, 0.1, 0.1, 0.2, 0.1]
        )[0]

        if mutation_type == 'extract_function' and len(lines) > 10:
            start = random.randint(2, len(lines) - 5)
            end = start + random.randint(2, 5)
            extracted = lines[start:end]
            func_name = f"extracted_func_{random.randint(1000, 9999)}"
            new_func = [f"def {func_name}():"] + ['    ' + line for line in extracted] + ['']
            lines[start:end] = [f"    {func_name}()"]
            lines = new_func + lines
        elif mutation_type == 'inline' and 'def ' in code:
            pass

        return '\n'.join(lines)

    def _mutate_semantic(self, code: str, context: Dict[str, Any]) -> str:
        """Mutación semántica basada en contexto."""
        has_loops = 'for ' in code or 'while ' in code
        has_conditions = 'if ' in code
        has_functions = 'def ' in code
        mutations = []

        if not has_loops and len(code.split('\n')) > 5:
            mutations.append(self._add_loop_optimization)
        if not has_conditions:
            mutations.append(self._add_error_handling)
        if has_functions:
            mutations.append(self._add_memoization)

        if mutations:
            mutation = random.choice(mutations)
            return mutation(code, context)
        return code

    def _add_loop_optimization(self, code: str, context: Dict[str, Any]) -> str:
        """Añade optimización de bucle."""
        return code

    def _add_error_handling(self, code: str, context: Dict[str, Any]) -> str:
        """Añade manejo de errores."""
        if 'try:' in code:
            return code
        lines = code.split('\n')
        indented = ['    ' + line for line in lines if line.strip()]
        wrapped = ['try:'] + indented + ['except Exception as e:', '    pass']
        return '\n'.join(wrapped)

    def _add_memoization(self, code: str, context: Dict[str, Any]) -> str:
        """Añade memoización a funciones puras."""
        if '@lru_cache' in code:
            return code
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and 'pure' in context.get('hints', []):
                lines.insert(i, '@lru_cache(maxsize=128)')
                break
        return '\n'.join(lines)

    def get_name(self) -> str:
        return "AdaptiveMutation"

    def update_success(self, success: bool):
        """Actualiza historial de éxito."""
        self.success_history.append(1 if success else 0)


class SemanticCrossover(GeneticOperator):
    """Crossover que preserva la semántica del código."""

    def apply(self, parent1: Dict[str, Any], parent2: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Aplica crossover semántico (retorna dos hijos)."""
        try:
            ast1 = ast.parse(parent1['code'])
            ast2 = ast.parse(parent2['code'])
        except SyntaxError:
            return self._simple_crossover(parent1, parent2)

        functions1 = [node for node in ast.walk(ast1) if isinstance(node, ast.FunctionDef)]
        functions2 = [node for node in ast.walk(ast2) if isinstance(node, ast.FunctionDef)]
        classes1 = [node for node in ast.walk(ast1) if isinstance(node, ast.ClassDef)]
        classes2 = [node for node in ast.walk(ast2) if isinstance(node, ast.ClassDef)]

        child1_ast = ast.Module(body=[])
        child2_ast = ast.Module(body=[])

        all_functions = functions1 + functions2
        random.shuffle(all_functions)
        mid = len(all_functions) // 2
        child1_ast.body.extend(all_functions[:mid])
        child2_ast.body.extend(all_functions[mid:])

        all_classes = classes1 + classes2
        random.shuffle(all_classes)
        mid = len(all_classes) // 2
        child1_ast.body.extend(all_classes[:mid])
        child2_ast.body.extend(all_classes[mid:])

        try:
            child1_code = ast.unparse(child1_ast)
            child2_code = ast.unparse(child2_ast)
        except Exception:
            child1_code = self._ast_to_code(child1_ast)
            child2_code = self._ast_to_code(child2_ast)

        m1 = list(parent1.get('mutations', []))
        m2 = list(parent2.get('mutations', []))
        child1 = {
            'code': child1_code,
            'fitness': 0.0,
            'age': 0,
            'mutations': m1 + m2 + ['semantic_crossover']
        }
        child2 = {
            'code': child2_code,
            'fitness': 0.0,
            'age': 0,
            'mutations': m2 + m1 + ['semantic_crossover']
        }
        return child1, child2

    def _simple_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover simple por líneas."""
        lines1 = parent1['code'].split('\n')
        lines2 = parent2['code'].split('\n')
        point = random.randint(1, min(len(lines1), len(lines2)) - 1) if min(len(lines1), len(lines2)) > 1 else 0
        child1_lines = lines1[:point] + lines2[point:]
        child2_lines = lines2[:point] + lines1[point:]
        child1 = {
            'code': '\n'.join(child1_lines),
            'fitness': 0.0,
            'age': 0,
            'mutations': list(parent1.get('mutations', [])) + ['line_crossover']
        }
        child2 = {
            'code': '\n'.join(child2_lines),
            'fitness': 0.0,
            'age': 0,
            'mutations': list(parent2.get('mutations', [])) + ['line_crossover']
        }
        return child1, child2

    def _ast_to_code(self, tree: ast.AST) -> str:
        """Convierte AST a código (fallback para Python < 3.9)."""
        return "# Generated code\npass"

    def get_name(self) -> str:
        return "SemanticCrossover"
