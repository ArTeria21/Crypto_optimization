import numpy as np
from typing import List, Tuple, Dict, Any
from ..core.base import MultiObjectiveOptimizer
from ..core.solution import Solution
import pandas as pd
import plotly.express as px
import random


class MOPSO(MultiObjectiveOptimizer):
    """
    Класс, реализующий алгоритм MOPSO для многокритериальной оптимизации.
    """

    def __init__(self,
                population_size: int,
                max_iterations: int,
                variable_bounds: List[Tuple[float, float]],
                objectives: List[Dict[str, Any]],
                inertia_weight: float = 0.729,  # Вес инерции
                cognitive_constant: float = 1.49445,  # Когнитивная компонента
                social_constant: float = 1.49445,     # Социальная компонента
                repository_size: int = 100) -> None:
        """
        Инициализация алгоритма MOPSO.

        Параметры:
        ----------
        population_size : int
            Размер роя (популяции).
        max_iterations : int
            Максимальное количество итераций.
        variable_bounds : List[Tuple[float, float]]
            Диапазоны для каждой переменной.
        objectives : List[Dict[str, Any]]
            Список целевых функций.
        inertia_weight : float
            Коэффициент инерции.
        cognitive_constant : float
            Коэффициент когнитивной компоненты.
        social_constant : float
            Коэффициент социальной компоненты.
        repository_size : int
            Максимальный размер репозитория (архива) решений.
        """
        super().__init__(variable_bounds, objectives, population_size)
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.repository_size = repository_size
        self.velocities: List[np.ndarray] = []
        self.personal_best_positions: List[np.ndarray] = []
        self.personal_best_values: List[np.ndarray] = []
        self.repository: List[Solution] = []

    def initialize_population(self) -> None:
        """
        Инициализирует популяцию частиц и их скорости.
        """
        lower_bounds = np.array([b[0] for b in self.variable_bounds])
        upper_bounds = np.array([b[1] for b in self.variable_bounds])
        dimension = self.num_variables

        for _ in range(self.population_size):
            variables = np.random.uniform(lower_bounds, upper_bounds)
            velocity = np.zeros(dimension)
            solution = Solution(variables=variables, objectives=self.objectives)
            self.population.append(solution)
            self.velocities.append(velocity)
            self.personal_best_positions.append(variables.copy())
            self.personal_best_values.append(solution.objective_values.copy())

        # Инициализируем репозиторий Парето-оптимальных решений
        self.update_repository()

    def evaluate_population(self) -> None:
        """
        Оценивает текущую популяцию по всем целевым функциям.
        """
        for solution in self.population:
            solution.objective_values = solution.evaluate_objectives()

    def update_personal_bests(self) -> None:
        """
        Обновляет личные лучшие позиции частиц.
        """
        for i, particle in enumerate(self.population):
            current_values = particle.objective_values
            personal_best_values = self.personal_best_values[i]
            if self.dominates(current_values, personal_best_values):
                self.personal_best_positions[i] = particle.variables.copy()
                self.personal_best_values[i] = current_values.copy()

    def dominates(self, values1: np.ndarray, values2: np.ndarray) -> bool:
        """
        Проверяет, доминирует ли первое решение над вторым.

        Параметры:
        ----------
        values1 : np.ndarray
            Значения целевых функций первого решения.
        values2 : np.ndarray
            Значения целевых функций второго решения.

        Возвращает:
        ----------
        bool
            True, если первое решение доминирует над вторым, иначе False.
        """
        better_in_all = True
        better_in_at_least_one = False

        for val1, val2, obj in zip(values1, values2, self.objectives):
            if obj['minimize']:
                if val1 > val2:
                    better_in_all = False
                elif val1 < val2:
                    better_in_at_least_one = True
            else:
                if val1 < val2:
                    better_in_all = False
                elif val1 > val2:
                    better_in_at_least_one = True

        return better_in_all and better_in_at_least_one

    def update_repository(self) -> None:
        """
        Обновляет репозиторий Парето-оптимальных решений.
        """
        combined = self.repository + self.population
        new_repository = []
        for solution in combined:
            dominated = False
            for other_solution in combined:
                if other_solution is not solution and other_solution.dominates(solution):
                    dominated = True
                    break
            if not dominated:
                new_repository.append(solution)

        # Ограничиваем размер репозитория
        if len(new_repository) > self.repository_size:
            self.repository = self.crowding_distance_selection(new_repository, self.repository_size)
        else:
            self.repository = new_repository

    def crowding_distance_selection(self, solutions: List[Solution], size: int) -> List[Solution]:
        """
        Отбирает решения по расстоянию сжатия для сохранения разнообразия.

        Параметры:
        ----------
        solutions : List[Solution]
            Список решений для отбора.
        size : int
            Требуемый размер выборки.

        Возвращает:
        ----------
        List[Solution]
            Отобранные решения.
        """
        # Вычисляем расстояния сжатия
        self.calculate_crowding_distance(solutions)
        # Сортируем по расстоянию сжатия
        solutions.sort(key=lambda x: x.crowding_distance, reverse=True)
        return solutions[:size]

    def calculate_crowding_distance(self, solutions: List[Solution]) -> None:
        """
        Вычисляет расстояния сжатия для заданного списка решений.

        Параметры:
        ----------
        solutions : List[Solution]
            Список решений для вычисления расстояний.
        """
        num_solutions = len(solutions)
        for solution in solutions:
            solution.crowding_distance = 0.0

        for m in range(self.num_objectives):
            solutions.sort(key=lambda x: x.objective_values[m])
            solutions[0].crowding_distance = solutions[-1].crowding_distance = float('inf')
            min_value = solutions[0].objective_values[m]
            max_value = solutions[-1].objective_values[m]
            if max_value - min_value == 0:
                continue
            for i in range(1, num_solutions - 1):
                solutions[i].crowding_distance += (
                    (solutions[i + 1].objective_values[m] - solutions[i - 1].objective_values[m]) /
                    (max_value - min_value)
                )

    def select_global_best(self) -> np.ndarray:
        """
        Выбирает глобально лучшее решение из репозитория.

        Возвращает:
        ----------
        np.ndarray
            Переменные глобально лучшего решения.
        """
        return random.choice(self.repository).variables.copy()

    def run(self) -> None:
        """
        Запускает алгоритм MOPSO.
        """
        self.initialize_population()
        self.evaluate_population()

        for iteration in range(self.max_iterations):
            print(f"Итерация {iteration + 1}/{self.max_iterations}")

            self.update_personal_bests()
            self.update_repository()

            for i in range(self.population_size):
                # Обновление скорости
                r1 = np.random.rand(self.num_variables)
                r2 = np.random.rand(self.num_variables)
                cognitive_velocity = self.cognitive_constant * r1 * (self.personal_best_positions[i] - self.population[i].variables)
                global_best_position = self.select_global_best()
                social_velocity = self.social_constant * r2 * (global_best_position - self.population[i].variables)
                self.velocities[i] = (
                    self.inertia_weight * self.velocities[i] +
                    cognitive_velocity +
                    social_velocity
                )
                # Обновление позиции
                self.population[i].variables += self.velocities[i]
                # Проверка границ
                for d in range(self.num_variables):
                    lower, upper = self.variable_bounds[d]
                    if self.population[i].variables[d] < lower:
                        self.population[i].variables[d] = lower
                        self.velocities[i][d] = 0
                    elif self.population[i].variables[d] > upper:
                        self.population[i].variables[d] = upper
                        self.velocities[i][d] = 0

            self.evaluate_population()

        print("Алгоритм MOPSO завершил работу.")

    def get_pareto_front(self) -> List[Solution]:
        """
        Возвращает найденный Парето-фронт из репозитория.

        Возвращает:
        ----------
        List[Solution]
            Список решений на Парето-фронте.
        """
        return self.repository

    def visualize_pareto_front(self) -> None:
        """
        Визуализирует Парето-фронт с помощью Plotly.
        """
        pareto_solutions = self.get_pareto_front()
        objective_values = np.array([solution.objective_values for solution in pareto_solutions])

        df = pd.DataFrame(objective_values, columns=[self.objectives[i]['name'] for i in range(self.num_objectives)])

        if self.num_objectives == 2:
            fig = px.scatter(df, x=self.objectives[0]['name'], y=self.objectives[1]['name'], title='Парето-фронт MOPSO')
        elif self.num_objectives == 3:
            fig = px.scatter_3d(df, x=self.objectives[0]['name'], y=self.objectives[1]['name'], z=self.objectives[2]['name'], title='Парето-фронт MOPSO')
        else:
            print("Визуализация доступна только для 2 или 3 целевых функций.")
            return
        fig.show()
