# solution.py

from typing import List, Dict, Any
import numpy as np


class Solution:
    """
    Класс, представляющий решение в пространстве переменных.
    """

    def __init__(self, variables: np.ndarray, objectives: List[Dict[str, Any]]) -> None:
        """
        Инициализация решения.

        Параметры:
        ----------
        variables : np.ndarray
            Массив значений переменных.
        objectives : List[Dict[str, Any]]
            Список словарей с информацией о целевых функциях.
        """
        self.variables: np.ndarray = variables  # Значения переменных
        self.objectives: List[Dict[str, Any]] = objectives  # Целевые функции
        self.objective_values: np.ndarray = self.evaluate_objectives()  # Значения целевых функций

    def evaluate_objectives(self) -> np.ndarray:
        """
        Вычисляет значения всех целевых функций для текущего решения.

        Возвращает:
        ----------
        np.ndarray
            Массив значений целевых функций.
        """
        values = np.array([obj['function'](self.variables) for obj in self.objectives])
        return values

    def dominates(self, other: 'Solution') -> bool:
        """
        Проверяет, доминирует ли текущее решение над другим по Парето.

        Параметры:
        ----------
        other : Solution
            Другое решение для сравнения.

        Возвращает:
        ----------
        bool
            True, если текущее решение доминирует над другим, иначе False.
        """
        better_in_all = True
        better_in_at_least_one = False

        for self_val, other_val, obj in zip(self.objective_values, other.objective_values, self.objectives):
            if obj['minimize']:
                if self_val > other_val:
                    better_in_all = False
                elif self_val < other_val:
                    better_in_at_least_one = True
            else:
                if self_val < other_val:
                    better_in_all = False
                elif self_val > other_val:
                    better_in_at_least_one = True

        return better_in_all and better_in_at_least_one

    def __str__(self) -> str:
        """
        Возвращает строковое представление решения.

        Возвращает:
        ----------
        str
            Строковое представление решения.
        """
        obj_str = ', '.join([f"{self.objectives[i]['name']}: {round(val, 4)}" for i, val in enumerate(self.objective_values)])
        var_str = ', '.join([f"x{i+1}: {round(val, 4)}" for i, val in enumerate(self.variables)])
        return f"Variables: [{var_str}] | Objectives: [{obj_str}]"
