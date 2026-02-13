"""
Motor de evolución de código con estrategias avanzadas.
Depende de: taec.core (perf_monitor, AdaptiveCache), operators, history, fitness_ml (opcional).
"""

import ast
import re
import hashlib
import random
import concurrent.futures
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

from taec.core.cache import AdaptiveCache
from taec.core.monitoring import perf_monitor

from taec.evolution.operators import AdaptiveMutation, SemanticCrossover
from taec.evolution.history import EvolutionHistory

try:
    from taec.evolution.fitness_ml import (
        TORCH_AVAILABLE,
        FitnessDataset,
        FitnessPredictor,
    )
    if not TORCH_AVAILABLE:
        FitnessDataset = None
        FitnessPredictor = None
except ImportError:
    TORCH_AVAILABLE = False
    FitnessDataset = None
    FitnessPredictor = None

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

if TORCH_AVAILABLE and FitnessDataset is not None and FitnessPredictor is not None:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader


class CodeEvolutionEngine:
    """Motor de evolución de código mejorado con estrategias avanzadas."""

    def __init__(self):
        self.population: List[Dict[str, Any]] = []
        self.population_size = 50
        self.elite_size = 5
        self.generation = 0
        self.fitness_history: List[Dict[str, Any]] = []
        self.best_solutions: List[Dict[str, Any]] = []

        self.mutation_operators = [AdaptiveMutation(0.15)]
        self.crossover_operator = SemanticCrossover()
        self.fitness_cache = AdaptiveCache[float](max_size=1000, ttl=300)

        if TORCH_AVAILABLE and FitnessDataset and FitnessPredictor:
            self.fitness_predictor = FitnessPredictor()
            self.fitness_dataset = FitnessDataset()
            self.train_predictor_every = 50
        else:
            self.fitness_predictor = None
            self.fitness_dataset = None

        self.diversity_threshold = 0.3
        self.innovation_archive: set = set()

        self.evolution_strategies = {
            'standard': self._standard_evolution,
            'island': self._island_evolution,
            'coevolution': self._coevolution,
            'novelty_search': self._novelty_search
        }
        self.current_strategy = 'standard'

    def _initialize_population(self, template: str, context: Dict[str, Any]):
        """Inicializa la población a partir del template."""
        self.population = []
        base = {
            'code': template,
            'fitness': 0.0,
            'age': 0,
            'mutations': ['initial']
        }
        self.population.append(base)
        while len(self.population) < self.population_size:
            if len(self.population) == 1:
                code_var = self._generate_random_variation(template)
            else:
                code_var = self._generate_random_variation(self.population[0]['code'])
            self.population.append({
                'code': code_var,
                'fitness': 0.0,
                'age': 0,
                'mutations': ['initial_variation']
            })

    def _generate_random_variation(self, template: str) -> str:
        """Genera variación aleatoria del template."""
        variations = [
            self._add_random_function,
            self._reorganize_code,
            self._change_algorithm,
            self._add_optimization
        ]
        return random.choice(variations)(template)

    def _add_random_function(self, code: str) -> str:
        functions = [
            "\ndef optimize_performance(data):\n    if isinstance(data, list):\n        return [x for x in data if x is not None]\n    return data\n",
            "\ndef validate_input(value, expected_type=None):\n    if expected_type and not isinstance(value, expected_type):\n        raise TypeError(f'Expected {expected_type}')\n    return value\n",
        ]
        return code + random.choice(functions)

    def _reorganize_code(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            imports, functions, main_code = [], [], []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(node)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node)
                else:
                    main_code.append(node)
            random.shuffle(functions)
            new_tree = ast.Module(body=imports + functions + main_code)
            if hasattr(ast, 'unparse'):
                return ast.unparse(new_tree)
        except Exception:
            pass
        return code

    def _change_algorithm(self, code: str) -> str:
        replacements = {
            r'for\s+(\w+)\s+in\s+range\(len\((\w+)\)\):': r'for i, _ in enumerate(\2):',
            r'if\s+(\w+)\s*==\s*True:': r'if \1:',
            r'if\s+(\w+)\s*==\s*False:': r'if not \1:',
        }
        for pattern, replacement in replacements.items():
            code = re.sub(pattern, replacement, code)
        return code

    def _add_optimization(self, code: str) -> str:
        optimizations = [
            (r'sum\(\[(.+) for (\w+) in (\w+)\]\)', r'sum(\1 for \2 in \3)'),
        ]
        for pattern, replacement in optimizations:
            code = re.sub(pattern, replacement, code, flags=re.MULTILINE)
        return code

    def evolve_code(self, template: str, context: Dict[str, Any],
                    generations: int = 100, strategy: str = 'standard') -> Tuple[str, float]:
        """Evoluciona código con estrategia seleccionada."""
        self.current_strategy = strategy
        with perf_monitor.timer("evolution_total"):
            self._initialize_population(template, context)
            if strategy in self.evolution_strategies:
                self.evolution_strategies[strategy](generations, context)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            best_idx = int(np.argmax([ind['fitness'] for ind in self.population]))
            best_solution = self.population[best_idx]
            self.best_solutions.append({
                'generation': self.generation,
                'code': best_solution['code'],
                'fitness': best_solution['fitness'],
                'strategy': strategy
            })
            return best_solution['code'], best_solution['fitness']

    def _standard_evolution(self, generations: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evolución estándar con mejoras."""
        gen = 0
        for gen in range(generations):
            self.generation = gen
            with perf_monitor.timer("generation"):
                fitness_scores = self._evaluate_population(context)
                best_fitness = max(fitness_scores)
                avg_fitness = float(np.mean(fitness_scores))
                diversity = self._calculate_diversity()
                self.fitness_history.append({
                    'generation': gen,
                    'best': best_fitness,
                    'average': avg_fitness,
                    'diversity': diversity
                })
                if gen % 10 == 0:
                    logger.info(
                        f"Generation {gen}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}, Diversity={diversity:.3f}"
                    )
                if best_fitness > 0.95:
                    break
                if self._is_stagnant():
                    self._apply_diversity_boost()
                new_population = self._selection(fitness_scores)
                self.population = self._reproduction(new_population, context)
                if self.fitness_predictor and TORCH_AVAILABLE and gen % self.train_predictor_every == 0:
                    self._train_fitness_predictor()
        return {'generations_run': gen + 1}

    def _island_evolution(self, generations: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Modelo de islas para evolución paralela."""
        n_islands = 4
        migration_interval = 10
        migration_size = 2
        island_size = self.population_size // n_islands
        islands = [self.population[i * island_size:(i + 1) * island_size] for i in range(n_islands)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_islands) as executor:
            for gen in range(generations):
                futures = []
                for island_idx, island in enumerate(islands):
                    futures.append(executor.submit(self._evolve_island, island, context, island_idx))
                islands = [f.result() for f in futures]
                if gen % migration_interval == 0 and gen > 0:
                    islands = self._migrate_between_islands(islands, migration_size)
                if gen % 10 == 0:
                    all_fitness = [ind['fitness'] for island in islands for ind in island]
                    logger.info(f"Island evolution gen {gen}: Best={max(all_fitness):.3f}, Avg={np.mean(all_fitness):.3f}")

        self.population = [ind for island in islands for ind in island]
        return {'generations_run': generations, 'islands': n_islands}

    def _evolve_island(self, island: List[Dict[str, Any]], context: Dict[str, Any], island_idx: int) -> List[Dict[str, Any]]:
        """Evoluciona una isla independientemente."""
        fitness_scores = []
        for individual in island:
            fitness = self._evaluate_fitness(individual['code'], context)
            individual['fitness'] = fitness
            fitness_scores.append(fitness)
        new_island = []
        elite_indices = np.argsort(fitness_scores)[-2:]
        for idx in elite_indices:
            new_island.append(island[idx].copy())
        while len(new_island) < len(island):
            parent1 = self._tournament_select(island, fitness_scores)
            parent2 = self._tournament_select(island, fitness_scores)
            if random.random() < 0.7:
                child1, child2 = self.crossover_operator.apply(parent1, parent2, context)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            mutation_op = random.choice(self.mutation_operators)
            child1 = mutation_op.apply(child1, context)
            child2 = mutation_op.apply(child2, context)
            new_island.extend([child1, child2])
        return new_island[:len(island)]

    def _migrate_between_islands(self, islands: List[List[Dict[str, Any]]], migration_size: int) -> List[List[Dict[str, Any]]]:
        """Migración entre islas."""
        n_islands = len(islands)
        for i in range(n_islands):
            source = islands[i]
            fitness_scores = [ind['fitness'] for ind in source]
            best_indices = np.argsort(fitness_scores)[-migration_size:]
            migrants = [source[idx].copy() for idx in best_indices]
            target_idx = (i + 1) % n_islands
            target = islands[target_idx]
            target_fitness = [ind['fitness'] for ind in target]
            worst_indices = np.argsort(target_fitness)[:migration_size]
            for j, idx in enumerate(worst_indices):
                target[idx] = migrants[j]
        return islands

    def _coevolution(self, generations: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Coevolución de múltiples poblaciones."""
        n_populations = 3
        specializations = ['performance', 'readability', 'correctness']
        populations = {spec: self.population[i::n_populations] for i, spec in enumerate(specializations)}

        for gen in range(generations):
            for spec, pop in populations.items():
                for individual in pop:
                    individual['fitness'] = self._evaluate_specialized_fitness(individual['code'], context, spec)
            if gen % 20 == 0:
                best_individuals = {}
                for spec, pop in populations.items():
                    best_idx = int(np.argmax([ind['fitness'] for ind in pop]))
                    best_individuals[spec] = pop[best_idx].copy()
                for spec, pop in populations.items():
                    for other_spec, best_ind in best_individuals.items():
                        if other_spec != spec:
                            worst_idx = int(np.argmin([ind['fitness'] for ind in pop]))
                            pop[worst_idx] = best_ind.copy()
            for spec, pop in populations.items():
                fitness_scores = [ind['fitness'] for ind in pop]
                new_pop = self._selection(fitness_scores, population=pop)
                populations[spec] = self._reproduction(new_pop, context)

        self.population = []
        for pop in populations.values():
            self.population.extend(pop)
        return {'generations_run': generations, 'populations': n_populations}

    def _novelty_search(self, generations: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Búsqueda por novedad."""
        novelty_archive = []
        k_nearest = 15
        for gen in range(generations):
            novelty_scores = []
            for individual in self.population:
                behavior = self._extract_behavior(individual['code'], context)
                distances = []
                for other in self.population:
                    if other != individual:
                        distances.append(self._behavior_distance(behavior, self._extract_behavior(other['code'], context)))
                for archived in novelty_archive:
                    distances.append(self._behavior_distance(behavior, archived))
                distances.sort()
                novelty = float(np.mean(distances[:k_nearest])) if distances else 0.0
                novelty_scores.append(novelty)
                individual['novelty'] = novelty
                individual['behavior'] = behavior
            threshold = float(np.percentile(novelty_scores, 90))
            for i, individual in enumerate(self.population):
                if novelty_scores[i] > threshold:
                    novelty_archive.append(individual['behavior'])
            if len(novelty_archive) > 500:
                novelty_archive = novelty_archive[-500:]
            if gen % 10 == 0:
                logger.info(f"Novelty search gen {gen}: Max novelty={max(novelty_scores):.3f}, Archive size={len(novelty_archive)}")
            new_population = self._selection(novelty_scores, selection_pressure=1.5)
            self.population = self._reproduction(new_population, context)
        for individual in self.population:
            individual['fitness'] = self._evaluate_fitness(individual['code'], context)
        return {'generations_run': generations, 'archive_size': len(novelty_archive)}

    def _extract_behavior(self, code: str, context: Dict[str, Any]) -> np.ndarray:
        """Extrae vector de comportamiento del código."""
        features = [len(code), code.count('\n'), code.count('def '), code.count('class '),
                    code.count('if '), code.count('for '), code.count('while ')]
        try:
            tree = ast.parse(code)
            features.append(self._ast_depth(tree))
            node_counts = defaultdict(int)
            for node in ast.walk(tree):
                node_counts[type(node).__name__] += 1
            for node_type in ['FunctionDef', 'Call', 'Name', 'Assign', 'If', 'For', 'While', 'Return', 'BinOp', 'Compare']:
                features.append(node_counts.get(node_type, 0))
        except Exception:
            features.extend([0] * 11)
        behavior = np.array(features, dtype=float)
        norm = np.linalg.norm(behavior)
        if norm > 0:
            behavior /= norm
        return behavior

    def _behavior_distance(self, b1: np.ndarray, b2: np.ndarray) -> float:
        return float(np.linalg.norm(b1 - b2))

    def _ast_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        if not hasattr(node, '_fields'):
            return current_depth
        max_depth = current_depth
        for field_name in node._fields:
            field_value = getattr(node, field_name, None)
            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, ast.AST):
                        max_depth = max(max_depth, self._ast_depth(item, current_depth + 1))
            elif isinstance(field_value, ast.AST):
                max_depth = max(max_depth, self._ast_depth(field_value, current_depth + 1))
        return max_depth

    def _evaluate_specialized_fitness(self, code: str, context: Dict[str, Any], specialization: str) -> float:
        """Evalúa fitness según especialización."""
        base_fitness = self._evaluate_fitness(code, context)
        if specialization == 'performance':
            complexity = self._calculate_complexity(code)
            base_fitness *= 1.2 if complexity < 5 else (0.8 if complexity > 15 else 1.0)
        elif specialization == 'readability':
            lines = code.split('\n')
            avg_len = np.mean([len(line) for line in lines]) if lines else 0
            if avg_len < 80:
                base_fitness *= 1.1
            if code.count('#') > len(lines) / 10:
                base_fitness *= 1.1
        elif specialization == 'correctness':
            if 'try:' in code:
                base_fitness *= 1.15
            if 'assert' in code:
                base_fitness *= 1.1
            if len(code) < 100:
                base_fitness *= 0.9
        return min(base_fitness, 1.0)

    def _calculate_diversity(self) -> float:
        """Calcula diversidad genética de la población."""
        if len(self.population) < 2:
            return 0.0
        distances = []
        n = min(20, len(self.population))
        for i in range(n):
            for j in range(i + 1, n):
                code1, code2 = self.population[i]['code'], self.population[j]['code']
                len_diff = abs(len(code1) - len(code2)) / max(len(code1), len(code2), 1)
                chars1, chars2 = set(code1), set(code2)
                char_diff = len(chars1.symmetric_difference(chars2)) / len(chars1.union(chars2)) if chars1 or chars2 else 0
                distances.append((len_diff + char_diff) / 2)
        return float(np.mean(distances)) if distances else 0.0

    def _is_stagnant(self, window: int = 20) -> bool:
        """Detecta si la evolución está estancada."""
        if len(self.fitness_history) < window:
            return False
        recent = self.fitness_history[-window:]
        best_values = [h['best'] for h in recent]
        return (best_values[-1] - best_values[0]) < 0.01

    def _apply_diversity_boost(self):
        """Aplica boost de diversidad cuando hay estancamiento."""
        logger.info("Applying diversity boost due to stagnation")
        for i in range(len(self.population) // 2, len(self.population)):
            individual = self.population[i].copy()
            if 'mutations' not in individual:
                individual['mutations'] = []
            for _ in range(3):
                individual = random.choice(self.mutation_operators).apply(individual, {})
            individual['mutations'].append('diversity_boost')
            self.population[i] = individual
        n_new = self.population_size // 10
        template = self.population[0]['code']
        for _ in range(n_new):
            new_individual = {
                'code': self._generate_random_variation(template),
                'fitness': 0.0,
                'age': 0,
                'mutations': ['random_injection']
            }
            idx = random.randint(self.elite_size, len(self.population) - 1)
            self.population[idx] = new_individual

    def _train_fitness_predictor(self):
        """Entrena el predictor de fitness con datos recientes."""
        if not self.fitness_predictor or not TORCH_AVAILABLE or not self.fitness_dataset:
            return
        for individual in self.population:
            if individual.get('fitness', 0) > 0:
                features = self._extract_code_features(individual['code'])
                self.fitness_dataset.add_sample(features, individual['fitness'])
        if len(self.fitness_dataset) < 100:
            return
        dataloader = DataLoader(self.fitness_dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.fitness_predictor.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        self.fitness_predictor.train()
        total_loss = 0.0
        for epoch in range(5):
            for features, targets in dataloader:
                optimizer.zero_grad()
                predictions = self.fitness_predictor(features).squeeze()
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        self.fitness_predictor.eval()
        logger.info(f"Fitness predictor trained, loss: {total_loss / max(len(dataloader), 1):.4f}")

    def _tournament_select(self, population: List[Dict[str, Any]], fitness_scores: List[float], tournament_size: int = 3) -> Dict[str, Any]:
        """Selección por torneo."""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[int(np.argmax(tournament_fitness))]
        return population[winner_idx]

    def _selection(self, fitness_scores: List[float], population: Optional[List[Dict[str, Any]]] = None,
                   selection_pressure: float = 2.0) -> List[Dict[str, Any]]:
        """Selección con presión ajustable."""
        if population is None:
            population = self.population
        new_population = []
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        while len(new_population) < self.population_size:
            tournament_size = max(2, int(len(population) * 0.1 * selection_pressure))
            selected = self._tournament_select(population, fitness_scores, tournament_size)
            new_population.append(selected.copy())
        return new_population

    def _reproduction(self, population: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reproducción con operadores adaptativos."""
        new_population = []
        for i in range(min(self.elite_size, len(population))):
            new_population.append(population[i])
        while len(new_population) < self.population_size:
            parent1 = random.choice(population[self.elite_size:])
            parent2 = random.choice(population[self.elite_size:])
            if random.random() < 0.7:
                child1, child2 = self.crossover_operator.apply(parent1, parent2, context)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            for child in [child1, child2]:
                if random.random() < 0.8:
                    mutation_op = random.choice(self.mutation_operators)
                    child = mutation_op.apply(child, context)
                    if hasattr(mutation_op, 'update_success'):
                        child_fitness = self._evaluate_fitness(child['code'], context)
                        mutation_op.update_success(child_fitness > parent1.get('fitness', 0))
                child['age'] = child.get('age', 0) + 1
                new_population.append(child)
                if len(new_population) >= self.population_size:
                    break
        return new_population[:self.population_size]

    def _evaluate_population(self, context: Dict[str, Any]) -> List[float]:
        """Evalúa fitness de toda la población."""
        fitness_scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures_list = []
            for individual in self.population:
                code_hash = hashlib.sha256(individual['code'].encode()).hexdigest()
                cached = self.fitness_cache.get(code_hash)
                if cached is not None:
                    fitness_scores.append(cached)
                    individual['fitness'] = cached
                else:
                    future = executor.submit(self._evaluate_fitness, individual['code'], context)
                    futures_list.append((future, individual, code_hash))
            for future, individual, code_hash in futures_list:
                fitness = future.result()
                fitness_scores.append(fitness)
                individual['fitness'] = fitness
                self.fitness_cache.put(code_hash, fitness)
        return fitness_scores

    def _evaluate_fitness(self, code: str, context: Dict[str, Any]) -> float:
        """Evalúa fitness de un código individual."""
        with perf_monitor.timer("fitness_evaluation"):
            fitness = 0.0
            ast_metrics = {}
            try:
                tree = ast.parse(code)
                fitness += 0.15
                ast_metrics = self._analyze_ast(tree)
                if ast_metrics.get('has_functions') and ast_metrics.get('has_error_handling'):
                    fitness += 0.05
            except SyntaxError:
                return fitness

            fitness += self._evaluate_complexity(code, ast_metrics) * 0.2
            fitness += self._evaluate_quality(code, ast_metrics) * 0.2
            fitness += self._evaluate_requirements(code, context, ast_metrics) * 0.3
            fitness += self._evaluate_innovation(code) * 0.1

            if self.fitness_predictor and TORCH_AVAILABLE:
                try:
                    features = self._extract_code_features(code)
                    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        predicted = self.fitness_predictor(features_tensor).item()
                    fitness = fitness * 0.8 + predicted * 0.2
                except Exception:
                    pass

            if 'async def' in code:
                fitness += 0.02
            if '@' in code and 'def' in code:
                fitness += 0.02
            if 'yield' in code:
                fitness += 0.02
            return min(fitness, 1.0)

    def _analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Análisis profundo del AST."""
        metrics = {
            'node_count': 0, 'max_depth': 0, 'has_functions': False, 'has_classes': False,
            'has_error_handling': False, 'has_loops': False, 'has_conditionals': False,
            'function_count': 0, 'class_count': 0, 'import_count': 0, 'unique_names': set(),
            'complexity_nodes': 0
        }
        for node in ast.walk(tree):
            metrics['node_count'] += 1
            if isinstance(node, ast.FunctionDef):
                metrics['has_functions'] = True
                metrics['function_count'] += 1
            elif isinstance(node, ast.ClassDef):
                metrics['has_classes'] = True
                metrics['class_count'] += 1
            elif isinstance(node, ast.Try):
                metrics['has_error_handling'] = True
            elif isinstance(node, (ast.For, ast.While)):
                metrics['has_loops'] = True
                metrics['complexity_nodes'] += 1
            elif isinstance(node, ast.If):
                metrics['has_conditionals'] = True
                metrics['complexity_nodes'] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['import_count'] += 1
            elif isinstance(node, ast.Name):
                metrics['unique_names'].add(node.id)
        metrics['max_depth'] = self._ast_depth(tree)
        metrics['unique_names'] = len(metrics['unique_names'])
        return metrics

    def _evaluate_complexity(self, code: str, ast_metrics: Dict[str, Any]) -> float:
        cyclomatic = 1 + ast_metrics.get('complexity_nodes', 0)
        if cyclomatic < 2:
            return 0.5
        if cyclomatic <= 10:
            return 1.0
        if cyclomatic <= 20:
            return 0.8
        return 0.5

    def _evaluate_quality(self, code: str, ast_metrics: Dict[str, Any]) -> float:
        score = 0.0
        lines = code.strip().split('\n')
        if 10 <= len(lines) <= 100:
            score += 0.3
        elif 5 <= len(lines) <= 200:
            score += 0.2
        if ast_metrics.get('function_count', 0) > 0:
            score += 0.2
        if ast_metrics.get('has_error_handling'):
            score += 0.2
        if '"""' in code or "'''" in code:
            score += 0.1
        if ast_metrics.get('unique_names', 0) > 5:
            score += 0.1
        if lines and max(len(line) for line in lines) < 100:
            score += 0.1
        return min(score, 1.0)

    def _evaluate_requirements(self, code: str, context: Dict[str, Any], ast_metrics: Dict[str, Any]) -> float:
        requirements_met, total_requirements = 0, 0
        for func_name in context.get('required_functions', []):
            total_requirements += 1
            if f"def {func_name}" in code:
                requirements_met += 1
        for class_name in context.get('required_classes', []):
            total_requirements += 1
            if f"class {class_name}" in code:
                requirements_met += 1
        for keyword in context.get('required_keywords', []):
            total_requirements += 1
            if keyword in code:
                requirements_met += 1
        for pattern in context.get('required_patterns', []):
            total_requirements += 1
            if re.search(pattern, code):
                requirements_met += 1
        score = requirements_met / total_requirements if total_requirements > 0 else 0.5
        if context.get('prefer_async') and 'async def' in code:
            score = min(score + 0.1, 1.0)
        if context.get('prefer_classes') and ast_metrics.get('has_classes'):
            score = min(score + 0.1, 1.0)
        return score

    def _evaluate_innovation(self, code: str) -> float:
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        if code_hash in self.innovation_archive:
            return 0.0
        if len(self.innovation_archive) < 500:
            self.innovation_archive.add(code_hash)
        return 1.0

    def _extract_code_features(self, code: str) -> List[float]:
        """Extrae características del código para ML."""
        features = [
            len(code) / 1000, code.count('\n') / 100, code.count('def ') / 10, code.count('class ') / 5,
            self._calculate_complexity(code) / 20, code.count('(') / 50, code.count('[') / 20, code.count('{') / 20,
            code.count('=') / 30, code.count('.') / 40, code.count('return') / 10, code.count('import') / 5,
            code.count('try:') / 5, code.count('except') / 5, code.count('if ') / 20, code.count('for ') / 10,
            code.count('while ') / 5, code.count('lambda') / 3, code.count('@') / 5, code.count('async') / 3
        ]
        lines = code.split('\n')
        if lines:
            features.append(sum(len(line) for line in lines) / len(lines) / 80)
        else:
            features.append(0)
        features.append(sum(1 for line in lines if line.strip().startswith('#')) / max(len(lines), 1))
        indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        features.append(np.mean(indent_levels) / 4 / 5 if indent_levels else 0)
        tokens = re.findall(r'\w+', code)
        features.append(len(set(tokens)) / max(len(tokens), 1))
        features.append((code.count('and') + code.count('or') + code.count('not')) / 20)
        while len(features) < 30:
            features.append(0.0)
        return features[:30]

    def _calculate_complexity(self, code: str) -> int:
        """Complejidad ciclomática aproximada."""
        complexity = 1
        complexity += code.count('if ') + code.count('elif ')
        complexity += code.count('for ') + code.count('while ')
        complexity += code.count('except')
        complexity += code.count(' and ') + code.count(' or ')
        complexity += len(re.findall(r'\[.+for.+in.+\]', code))
        complexity += len(re.findall(r'\{.+for.+in.+\}', code))
        return complexity
