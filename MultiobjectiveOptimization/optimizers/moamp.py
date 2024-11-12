import numpy as np
from typing import List, Tuple, Dict, Any
from ..core.base import MultiObjectiveOptimizer
from ..core.solution import Solution
import pandas as pd
import plotly.express as px
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor  # Добавлено для параллельных вычислений соседей


def tabu_search_make_step(args):
    """
    Вспомогательная функция для вызова make_step на экземпляре MultiobjectiveTabuSearch.
    """
    tabu_search, metric_index = args
    tabu_search.make_step(metric_index)
    return tabu_search


def evaluate_neighbor(args):
    """
    Вспомогательная функция для оценки целевых функций соседа.
    """
    neighbor, objectives = args
    neighbor.objective_values = neighbor.evaluate_objectives()
    return neighbor


class MultiobjectiveTabuSearch:
    """
    Класс, реализующий многоцелевой табу-поиск для одного решения.
    """

    def __init__(self,
                variable_bounds: List[Tuple[float, float]],
                objectives: List[Dict[str, Any]],
                step_size: float,
                tabu_length: int) -> None:
        """
        Инициализация табу-поиска.
        """
        self.variable_bounds = variable_bounds
        self.objectives = objectives
        self.step_size = step_size
        self.tabu_length = tabu_length
        self.tabu_list: List[np.ndarray] = []
        self.current_solution: Solution = None
        self.best_solution: Solution = None

    def init_starting_point(self, solution: Solution) -> None:
        """
        Устанавливает начальное решение для табу-поиска.
        """
        self.current_solution = solution
        self.best_solution = solution

    def find_neighbors(self) -> List[Solution]:
        """
        Находит соседние решения путем изменения переменных на величину шага.
        """
        variations = []
        for var, (lower, upper) in zip(self.current_solution.variables, self.variable_bounds):
            steps = [var - self.step_size, var, var + self.step_size]
            steps = [s for s in steps if lower <= s <= upper]
            variations.append(steps)

        neighbor_vars = np.array(np.meshgrid(*variations)).T.reshape(-1, len(self.variable_bounds))
        neighbors = [Solution(variables=vars, objectives=self.objectives) for vars in neighbor_vars]
        return neighbors

    def get_best_neighbor(self, metric_index: int) -> Solution:
        """
        Находит лучшего соседа по заданной целевой функции.
        """
        neighbors = self.find_neighbors()
        neighbors = [n for n in neighbors if not any(np.array_equal(n.variables, t) for t in self.tabu_list)]

        if not neighbors:
            return self.current_solution

        # Параллельная оценка целевых функций соседей
        with ThreadPoolExecutor() as executor:
            neighbors = list(executor.map(
                evaluate_neighbor,
                [(neighbor, self.objectives) for neighbor in neighbors]
            ))

        obj = self.objectives[metric_index]
        minimize = obj['minimize']

        # Сортировка соседей по значению целевой функции
        neighbors.sort(key=lambda x: x.objective_values[metric_index], reverse=not minimize)
        best_neighbor = neighbors[0]
        return best_neighbor

    def make_step(self, metric_index: int) -> Solution:
        """
        Выполняет шаг табу-поиска.
        """
        self.tabu_list.append(self.current_solution.variables)
        if len(self.tabu_list) > self.tabu_length:
            self.tabu_list.pop(0)

        next_solution = self.get_best_neighbor(metric_index)
        self.current_solution = next_solution

        # Обновляем лучшее найденное решение
        if self.best_solution is None or self.current_solution.dominates(self.best_solution):
            self.best_solution = self.current_solution

        return self.current_solution


class MOAMP(MultiObjectiveOptimizer):
    """
    Класс, реализующий алгоритм MOAMP для многокритериальной оптимизации.
    """

    def __init__(self,
                variable_bounds: List[Tuple[float, float]],
                objectives: List[Dict[str, Any]],
                population_size: int = 30,
                step_size: float = 1.0,
                tabu_length: int = 10) -> None:
        """
        Инициализация MOAMP.
        """
        super().__init__(variable_bounds, objectives, population_size)
        self.step_size = step_size
        self.tabu_length = tabu_length
        self.tabu_searches: List[MultiobjectiveTabuSearch] = []
        self.history: List[List[Solution]] = []

    def initialize_population(self) -> None:
        """
        Инициализирует популяцию случайными решениями и запускает табу-поиски.
        """
        lower_bounds = np.array([b[0] for b in self.variable_bounds])
        upper_bounds = np.array([b[1] for b in self.variable_bounds])

        for _ in range(self.population_size):
            variables = np.random.uniform(lower_bounds, upper_bounds)
            solution = Solution(variables=variables, objectives=self.objectives)
            self.population.append(solution)

            tabu_search = MultiobjectiveTabuSearch(
                variable_bounds=self.variable_bounds,
                objectives=self.objectives,
                step_size=self.step_size,
                tabu_length=self.tabu_length
            )
            tabu_search.init_starting_point(solution)
            self.tabu_searches.append(tabu_search)

        self.update_pareto_front()

    def evaluate_population(self) -> None:
        """
        Оценивает популяцию решений по целевым функциям.
        """
        for solution in self.population:
            solution.objective_values = solution.evaluate_objectives()

    def run(self, iterations: int) -> None:
        """
        Запускает алгоритм MOAMP.
        """
        self.initialize_population()

        num_objectives = self.num_objectives
        for i in tqdm(range(iterations)):
            self.history.append(self.pareto_front.copy())
            metric_index = i % num_objectives
            self.make_step(metric_index)
            self.update_pareto_front()

        print("Алгоритм MOAMP завершил работу.")

    def make_step(self, metric_index: int) -> None:
        """
        Выполняет шаг оптимизации для всей популяции по заданной метрике.
        """
        # Параллельное выполнение табу-поисков
        with multiprocessing.Pool() as pool:
            self.tabu_searches = pool.map(
                tabu_search_make_step,
                [(tabu_search, metric_index) for tabu_search in self.tabu_searches]
            )
        self.population = [tabu_search.current_solution for tabu_search in self.tabu_searches]

    def update_pareto_front(self) -> None:
        """
        Обновляет Парето-фронт текущей популяции.
        """
        combined = self.population + self.pareto_front
        new_pareto_front = []
        for solution in combined:
            dominated = False
            for other_solution in combined:
                if other_solution is not solution and other_solution.dominates(solution):
                    dominated = True
                    break
            if not dominated and solution not in new_pareto_front:
                new_pareto_front.append(solution)
        self.pareto_front = new_pareto_front

    def get_pareto_front(self) -> List[Solution]:
        """
        Возвращает текущий Парето-фронт.
        """
        return self.pareto_front

    def visualize_pareto_front(self) -> None:
        """
        Визуализирует эволюцию Парето-фронта с помощью Plotly.
        """
        data = []
        for i, pareto_front in enumerate(self.history):
            for solution in pareto_front:
                entry = {'Iteration': i + 1}
                for idx, val in enumerate(solution.objective_values):
                    entry[self.objectives[idx]['name']] = val
                data.append(entry)

        df = pd.DataFrame(data)

        if self.num_objectives == 2:
            fig = px.scatter(df, x=self.objectives[0]['name'], y=self.objectives[1]['name'], color='Iteration',
                            title='Эволюция Парето-фронта')
        elif self.num_objectives == 3:
            fig = px.scatter_3d(df, x=self.objectives[0]['name'], y=self.objectives[1]['name'], z=self.objectives[2]['name'],
                                color='Iteration', title='Эволюция Парето-фронта')
        else:
            print("Визуализация доступна только для 2 или 3 целевых функций.")
            return
        fig.show()

    def print_pareto_front(self) -> None:
        """
        Выводит текущий Парето-фронт.
        """
        print("Текущий Парето-фронт:")
        for solution in self.pareto_front:
            print(solution)
