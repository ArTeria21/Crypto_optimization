import numpy as np
from typing import List, Tuple, Dict, Any
from ..core.base import MultiObjectiveOptimizer
from ..core.solution import Solution
import pandas as pd
import plotly.express as px


class NSGA_II(MultiObjectiveOptimizer):
    """
    Класс, реализующий алгоритм NSGA-II для многокритериальной оптимизации.
    """

    def __init__(self,
                 population_size: int,
                 max_generations: int,
                 variable_bounds: List[Tuple[float, float]],
                 objectives: List[Dict[str, Any]],
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9,
                 normalize_values: bool = False) -> None:
        """
        Инициализация алгоритма NSGA-II.

        Параметры:
        ----------
        population_size : int
            Размер популяции.
        max_generations : int
            Максимальное количество поколений.
        variable_bounds : List[Tuple[float, float]]
            Диапазоны для каждой переменной.
        objectives : List[Dict[str, Any]]
            Список целевых функций.
        mutation_rate : float
            Вероятность мутации.
        crossover_rate : float
            Вероятность кроссовера.
        normalize_values : bool
            Флаг для нормализации переменных.
        """
        super().__init__(variable_bounds, objectives, population_size)
        self.max_generations: int = max_generations
        self.mutation_rate: float = mutation_rate
        self.crossover_rate: float = crossover_rate
        self.normalize_values: bool = normalize_values  # Добавлено
        self.population: List[Solution] = []

    def initialize_population(self) -> None:
        """
        Инициализирует популяцию случайными решениями.
        """
        lower_bounds = np.array([b[0] for b in self.variable_bounds])
        upper_bounds = np.array([b[1] for b in self.variable_bounds])
        for _ in range(self.population_size):
            variables = np.random.uniform(lower_bounds, upper_bounds)
            solution = Solution(variables=variables, objectives=self.objectives, normalize_values=self.normalize_values)
            self.population.append(solution)

    def evaluate_population(self) -> None:
        """
        Оценивает текущую популяцию по всем целевым функциям.
        """
        for solution in self.population:
            solution.objective_values = solution.evaluate_objectives()

    def run(self) -> None:
        """
        Запускает алгоритм NSGA-II.
        """
        self.initialize_population()
        self.evaluate_population()

        for generation in range(self.max_generations):
            print(f"Поколение {generation + 1}/{self.max_generations}")

            offspring = self.create_offspring()
            self.evaluate_offspring(offspring)
            combined_population = self.population + offspring
            fronts = self.fast_non_dominated_sort(combined_population)
            new_population = []

            for front in fronts:
                self.calculate_crowding_distance(front)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_population.extend(front)
                if len(new_population) >= self.population_size:
                    break

            self.population = new_population[:self.population_size]

        print("Алгоритм NSGA-II завершил работу.")

    def create_offspring(self) -> List[Solution]:
        """
        Создает потомков путем применения кроссовера и мутации.

        Возвращает:
        ----------
        List[Solution]
            Список потомков.
        """
        offspring = []
        while len(offspring) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child_variables = self.crossover(parent1.variables, parent2.variables)
            child_variables = self.mutate(child_variables)
            child = Solution(variables=child_variables, objectives=self.objectives, normalize_values=self.normalize_values)
            offspring.append(child)
        return offspring

    def evaluate_offspring(self, offspring: List[Solution]) -> None:
        """
        Оценивает потомков по целевым функциям.

        Параметры:
        ----------
        offspring : List[Solution]
            Список потомков для оценки.
        """
        for child in offspring:
            child.objective_values = child.evaluate_objectives()

    def tournament_selection(self, k: int = 3) -> Solution:
        """
        Выполняет турнирный отбор.

        Параметры:
        ----------
        k : int
            Число участников турнира.

        Возвращает:
        ----------
        Solution
            Победитель турнира.
        """
        participants = np.random.choice(self.population, k)
        best = participants[0]
        for participant in participants[1:]:
            if participant.dominates(best):
                best = participant
            elif not best.dominates(participant) and np.random.rand() < 0.5:
                best = participant
        return best

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Выполняет одноточечный кроссовер между двумя родителями.

        Параметры:
        ----------
        parent1 : np.ndarray
            Первый родитель.
        parent2 : np.ndarray
            Второй родитель.

        Возвращает:
        ----------
        np.ndarray
            Массив переменных потомка.
        """
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.num_variables)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            return child
        else:
            return parent1.copy()

    def mutate(self, variables: np.ndarray) -> np.ndarray:
        """
        Выполняет мутацию над переменными решения.

        Параметры:
        ----------
        variables : np.ndarray
            Массив переменных для мутации.

        Возвращает:
        ----------
        np.ndarray
            Мутированный массив переменных.
        """
        for i in range(self.num_variables):
            if np.random.rand() < self.mutation_rate:
                lower, upper = self.variable_bounds[i]
                variables[i] = np.random.uniform(lower, upper)
        return variables

    def fast_non_dominated_sort(self, population: List[Solution]) -> List[List[Solution]]:
        """
        Выполняет быструю не доминирующую сортировку.

        Параметры:
        ----------
        population : List[Solution]
            Популяция для сортировки.

        Возвращает:
        ----------
        List[List[Solution]]
            Список фронтов, каждый из которых является списком решений.
        """
        fronts = []
        S = {}
        n = {}
        rank = {}

        for p in population:
            S[p] = []
            n[p] = 0
            for q in population:
                if p.dominates(q):
                    S[p].append(q)
                elif q.dominates(p):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                if len(fronts) == 0:
                    fronts.append([])
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def calculate_crowding_distance(self, front: List[Solution]) -> None:
        """
        Вычисляет расстояния сжатия для заданного фронта.

        Параметры:
        ----------
        front : List[Solution]
            Фронт для вычисления расстояний.
        """
        num_solutions = len(front)
        for solution in front:
            solution.crowding_distance = 0.0

        for m in range(self.num_objectives):
            front.sort(key=lambda x: x.objective_values[m])
            front[0].crowding_distance = front[-1].crowding_distance = float('inf')
            min_value = front[0].objective_values[m]
            max_value = front[-1].objective_values[m]
            if max_value - min_value == 0:
                continue
            for i in range(1, num_solutions - 1):
                front[i].crowding_distance += (
                    (front[i + 1].objective_values[m] - front[i - 1].objective_values[m]) /
                    (max_value - min_value)
                )

    def get_pareto_front(self) -> List[Solution]:
        """
        Возвращает найденный Парето-фронт.

        Возвращает:
        ----------
        List[Solution]
            Список решений на Парето-фронте.
        """
        fronts = self.fast_non_dominated_sort(self.population)
        pareto_front = fronts[0]
        return pareto_front

    def print_pareto_front(self) -> None:
        """
        Выводит текущий Парето-фронт.
        """
        pareto_front = self.get_pareto_front()
        print("Текущий Парето-фронт:")
        for solution in pareto_front:
            print(solution)

    def visualize_pareto_front(self) -> None:
        """
        Визуализирует Парето-фронт с помощью Plotly.
        """
        pareto_front = self.get_pareto_front()
        if not pareto_front:
            print("Парето-фронт пуст.")
            return

        objective_values = np.array([solution.objective_values for solution in pareto_front])

        df = pd.DataFrame(objective_values, columns=[self.objectives[i]['name'] for i in range(self.num_objectives)])

        if self.num_objectives == 2:
            fig = px.scatter(df, x=self.objectives[0]['name'], y=self.objectives[1]['name'], title='Парето-фронт')
        elif self.num_objectives == 3:
            fig = px.scatter_3d(df, x=self.objectives[0]['name'], y=self.objectives[1]['name'],
                                z=self.objectives[2]['name'], title='Парето-фронт')
        else:
            print("Визуализация доступна только для 2 или 3 целевых функций.")
            return
        fig.show()
