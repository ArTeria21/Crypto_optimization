import numpy as np
from typing import List, Tuple, Dict, Any
from ..core.base import MultiObjectiveOptimizer
from ..core.solution import Solution
import pandas as pd
import plotly.express as px
import math
import copy

class MOEA_D(MultiObjectiveOptimizer):
    """
    Класс, реализующий алгоритм MOEA/D для многокритериальной оптимизации.
    """

    def __init__(self,
                population_size: int,
                max_generations: int,
                variable_bounds: List[Tuple[float, float]],
                objectives: List[Dict[str, Any]],
                neighborhood_size: int = None,
                nr: int = 2,
                delta: float = 0.9,
                mutation_rate: float = 0.1,
                crossover_rate: float = 0.9,
                normalize_values: bool = False) -> None:
        """
        Инициализация алгоритма MOEA/D.
        """
        super().__init__(variable_bounds, objectives, population_size)
        self.max_generations: int = max_generations
        self.mutation_rate: float = mutation_rate
        self.crossover_rate: float = crossover_rate

        self.neighborhood_size: int = neighborhood_size if neighborhood_size is not None else int(math.sqrt(self.population_size))
        self.nr: int = nr
        self.delta: float = delta

        self.weight_vectors: np.ndarray = None
        self.neighbors: List[List[int]] = []
        self.z: np.ndarray = None  # Справочная точка (идеальная точка)
        self.archive: List[Solution] = []  # Архив недоминируемых решений
        self.normalize_values = normalize_values  # Добавлено

    def initialize_population(self) -> None:
        """
        Инициализирует популяцию случайными решениями.
        """
        self.population = []
        lower_bounds = np.array([b[0] for b in self.variable_bounds])
        upper_bounds = np.array([b[1] for b in self.variable_bounds])
        for _ in range(self.population_size):
            variables = np.random.uniform(lower_bounds, upper_bounds)
            solution = Solution(variables=variables, objectives=self.objectives, normalize_values=self.normalize_values)
            solution.objective_values = solution.evaluate_objectives()
            self.population.append(solution)

    def initialize_weight_vectors(self) -> None:
        """
        Инициализирует векторы весов для декомпозиции задачи.
        """
        num_objectives = self.num_objectives
        H = self.population_size - 1  # Параметр для генерации векторов весов
        self.weight_vectors = []

        if num_objectives == 2:
            for i in range(self.population_size):
                w1 = i / H
                w2 = 1 - w1
                self.weight_vectors.append([w1, w2])
        else:
            # Генерация весовых векторов для любого количества целей
            def uniform_points(n_samples, n_obj):
                def recurse(arr, left, total, n):
                    if n == 0:
                        if left == 0:
                            arr.append(total.copy())
                    else:
                        for i in range(left + 1):
                            total[n_obj - n] = i
                            recurse(arr, left - i, total, n - 1)
                arr = []
                total = [0] * n_obj
                recurse(arr, n_samples, total, n_obj)
                return np.array(arr) / n_samples

            self.weight_vectors = uniform_points(H, num_objectives)

        self.weight_vectors = np.array(self.weight_vectors)
        # Если векторов меньше, чем популяция, заполним оставшиеся случайными
        while len(self.weight_vectors) < self.population_size:
            random_weights = np.random.rand(self.num_objectives)
            random_weights /= np.sum(random_weights)
            self.weight_vectors = np.vstack([self.weight_vectors, random_weights])

        # Обрежем до размера популяции
        self.weight_vectors = self.weight_vectors[:self.population_size]

    def initialize_neighbors(self) -> None:
        """
        Инициализирует соседей для каждого вектора весов.
        """
        distances = np.zeros((self.population_size, self.population_size))
        for i in range(self.population_size):
            for j in range(self.population_size):
                distances[i, j] = np.linalg.norm(self.weight_vectors[i] - self.weight_vectors[j])
        for i in range(self.population_size):
            indices = np.argsort(distances[i])[:self.neighborhood_size]
            self.neighbors.append(indices.tolist())

    def initialize_reference_point(self) -> None:
        """
        Инициализирует справочную точку z* (идеальная точка).
        """
        self.z = np.full(self.num_objectives, np.inf)
        for solution in self.population:
            self.z = np.minimum(self.z, solution.objective_values)

    def evaluate_population(self) -> None:
        """
        Оценивает текущую популяцию по всем целевым функциям.
        """
        for solution in self.population:
            solution.objective_values = solution.evaluate_objectives()

    def run(self) -> None:
        """
        Запускает алгоритм MOEA/D.
        """
        self.initialize_weight_vectors()
        self.initialize_neighbors()
        self.initialize_population()
        self.evaluate_population()
        self.initialize_reference_point()

        for generation in range(self.max_generations):
            print(f"Поколение {generation + 1}/{self.max_generations}")
            offspring_population = []

            for i in range(self.population_size):
                # Шаг 1: Выбор родителей
                if np.random.rand() < self.delta:
                    mating_pool = self.neighbors[i]
                else:
                    mating_pool = list(range(self.population_size))
                parents_indices = np.random.choice(mating_pool, 2, replace=False)
                parent1 = self.population[parents_indices[0]]
                parent2 = self.population[parents_indices[1]]

                # Шаг 2: Генерация потомка
                child_variables = self.sbx_crossover(parent1.variables, parent2.variables)
                child_variables = self.polynomial_mutation(child_variables)
                child = Solution(variables=child_variables, objectives=self.objectives, normalize_values=self.normalize_values)
                child.objective_values = child.evaluate_objectives()

                # Шаг 3: Обновление справочной точки
                self.update_reference_point(child)

                # Шаг 4: Обновление решений
                self.update_subproblem(child, i)

                # Шаг 5: Обновление архива
                self.update_archive(child)

        print("Алгоритм MOEA/D завершил работу.")

    def sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray, eta: float = 20) -> np.ndarray:
        """
        Выполняет SBX (Simulated Binary Crossover) между двумя родителями.
        """
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            if np.random.rand() <= self.crossover_rate:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    x1 = min(parent1[i], parent2[i])
                    x2 = max(parent1[i], parent2[i])
                    lower, upper = self.variable_bounds[i]
                    rand = np.random.rand()
                    beta = 1.0 + (2.0 * (x1 - lower) / (x2 - x1))
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
                    beta = 1.0 + (2.0 * (upper - x2) / (x2 - x1))
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
                    c1 = np.clip(c1, lower, upper)
                    c2 = np.clip(c2, lower, upper)
                    if np.random.rand() < 0.5:
                        child[i] = c1
                    else:
                        child[i] = c2
                else:
                    child[i] = parent1[i]
            else:
                child[i] = parent1[i]
        return child

    def polynomial_mutation(self, variables: np.ndarray, eta: float = 20) -> np.ndarray:
        """
        Выполняет полиномиальную мутацию над переменными решения.
        """
        for i in range(len(variables)):
            if np.random.rand() < self.mutation_rate:
                x = variables[i]
                lower, upper = self.variable_bounds[i]
                delta1 = (x - lower) / (upper - lower)
                delta2 = (upper - x) / (upper - lower)
                rand = np.random.rand()
                mut_pow = 1.0 / (eta + 1.0)
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                    deltaq = 1.0 - val ** mut_pow
                x = x + deltaq * (upper - lower)
                x = np.clip(x, lower, upper)
                variables[i] = x
        return variables

    def update_reference_point(self, solution: Solution) -> None:
        """
        Обновляет справочную точку z* на основе нового решения.
        """
        self.z = np.minimum(self.z, solution.objective_values)

    def update_subproblem(self, child: Solution, index: int) -> None:
        """
        Обновляет решение текущей подзадачи и соседей.
        """
        neighbor_indices = self.neighbors[index]
        for neighbor_index in neighbor_indices:
            weight_vector = self.weight_vectors[neighbor_index]
            current_solution = self.population[neighbor_index]
            f_current = self.tchebycheff(current_solution, weight_vector)
            f_child = self.tchebycheff(child, weight_vector)
            if f_child < f_current:
                self.population[neighbor_index] = Solution(child.variables.copy(), self.objectives, normalize_values=self.normalize_values)
                self.population[neighbor_index].objective_values = child.objective_values.copy()

    def update_archive(self, solution: Solution) -> None:
        """
        Обновляет архив недоминируемых решений.
        """
        non_dominated = True
        to_remove = []
        for archived_solution in self.archive:
            if archived_solution.dominates(solution):
                non_dominated = False
                break
            elif solution.dominates(archived_solution):
                to_remove.append(archived_solution)

        if non_dominated:
            self.archive.append(solution)
            for sol in to_remove:
                self.archive.remove(sol)

    def tchebycheff(self, solution: Solution, weight_vector: np.ndarray) -> float:
        """
        Вычисляет функцию Чебышёва для решения.
        """
        diff = np.abs(solution.objective_values - self.z)
        return np.max(weight_vector * diff)

    def get_pareto_front(self) -> List[Solution]:
        """
        Возвращает найденный Парето-фронт из архива.
        """
        return self.archive

    def print_pareto_front(self) -> None:
        """
        Выводит текущий Парето-фронт.
        """
        pareto_solutions = self.get_pareto_front()
        print("Текущий Парето-фронт:")
        for solution in pareto_solutions:
            print(solution)

    def visualize_pareto_front(self) -> None:
        """
        Визуализирует Парето-фронт с помощью Plotly.
        """
        pareto_solutions = self.get_pareto_front()
        if not pareto_solutions:
            print("Парето-фронт пуст.")
            return

        objective_values = np.array([solution.objective_values for solution in pareto_solutions])

        df = pd.DataFrame(objective_values, columns=[self.objectives[i]['name'] for i in range(self.num_objectives)])

        if self.num_objectives == 2:
            fig = px.scatter(df, x=self.objectives[0]['name'], y=self.objectives[1]['name'], title='Парето-фронт')
        elif self.num_objectives == 3:
            fig = px.scatter_3d(df, x=self.objectives[0]['name'], y=self.objectives[1]['name'], z=self.objectives[2]['name'], title='Парето-фронт')
        else:
            print("Визуализация доступна только для 2 или 3 целевых функций.")
            return
        fig.show()
