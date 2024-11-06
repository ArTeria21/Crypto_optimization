from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
from .solution import Solution


class MultiObjectiveOptimizer(ABC):
    """
    Абстрактный базовый класс для алгоритмов многокритериальной оптимизации.
    """

    def __init__(self,
                variable_bounds: List[Tuple[float, float]],
                objectives: List[Dict[str, Any]],
                population_size: int) -> None:
        """
        Инициализация оптимизатора.

        Параметры:
        ----------
        variable_bounds : List[Tuple[float, float]]
            Диапазоны для каждой переменной в виде списка кортежей (min, max).
        objectives : List[Dict[str, Any, str]]
            Список словарей с функциями и индикаторами минимизации, а также названием метрики.
        population_size : int
            Размер популяции.
        """
        self.variable_bounds: List[Tuple[float, float]] = variable_bounds
        self.objectives: List[Dict[str, Any, str]] = objectives
        self.population_size: int = population_size
        self.num_variables: int = len(variable_bounds)
        self.num_objectives: int = len(objectives)
        self.population: List[Solution] = []
        self.pareto_front: List[Solution] = []

    @abstractmethod
    def initialize_population(self) -> None:
        """
        Инициализирует популяцию решений.
        """
        pass

    @abstractmethod
    def evaluate_population(self) -> None:
        """
        Оценивает популяцию решений по целевым функциям.
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """
        Запускает алгоритм оптимизации.
        """
        pass

    def get_pareto_front(self) -> List[Solution]:
        """
        Получает Парето-фронт из текущей популяции.

        Возвращает:
        ----------
        List[Solution]
            Список решений на Парето-фронте.
        """
        pareto_front = []
        for solution in self.population:
            dominated = False
            for other_solution in self.population:
                if other_solution is not solution and other_solution.dominates(solution):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(solution)
        self.pareto_front = pareto_front
        return pareto_front

    def visualize_pareto_front(self) -> None:
        """
        Визуализирует Парето-фронт с помощью Plotly.
        """
        pareto_solutions = self.get_pareto_front()
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

    def print_pareto_front(self) -> None:
        """
        Выводит текущий Парето-фронт.
        """
        pareto_solutions = self.get_pareto_front()
        print("Текущий Парето-фронт:")
        for solution in pareto_solutions:
            print(solution)
