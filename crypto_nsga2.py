import numpy as np
import pandas as pd
import yfinance as yf
from MultiobjectiveOptimization.optimizers.nsga2 import NSGA_II
from MultiobjectiveOptimization.core.solution import Solution
from MultiobjectiveOptimization.core.base import MultiObjectiveOptimizer

# Шаг 1: Определение тикеров криптовалют
TICKERS = ["TON11419-USD", "SUI20947-USD",'ETH-USD', "LINK-USD", "APT21794-USD", "FET-USD", "TAO22974-USD"]

# Шаг 2: Загрузка исторических данных цен
data = yf.download(TICKERS, start="2024-09-01", end="2024-11-01", progress=False)["Adj Close"]

# Шаг 3: Вычисление дневной доходности
returns = data.pct_change().dropna()

# Шаг 4: Вычисление ожидаемой доходности и ковариационной матрицы
expected_returns = returns.mean()
cov_matrix = returns.cov()

# Количество активов
num_assets = len(TICKERS)

# Шаг 5: Определение границ переменных (0, 1) для каждой криптовалюты
variable_bounds = [(0.0, 1.0) for _ in range(num_assets)]

# Шаг 6: Определение функций цели

# Функция прибыли (для максимизации)
def sharpe_ratio_function(weights):
    weights = np.array(weights)
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0
    weights = weights / total_weight
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    risk_free_rate = 0.01  # Предположим безрисковую ставку 1%
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
    return sharpe_ratio  # Максимизируем коэффициент Шарпа


# Функция риска (для минимизации)
def var_function(weights):
    from scipy.stats import norm
    weights = np.array(weights)
    total_weight = np.sum(weights)
    if total_weight == 0:
        return np.inf
    weights = weights / total_weight
    portfolio_mean = np.sum(expected_returns * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    confidence_level = 0.95
    z_score = norm.ppf(1 - confidence_level)
    var = -(portfolio_mean + z_score * portfolio_std)
    return var  # Минимизируем VaR


# Список целевых функций
objectives = [
    {'function': sharpe_ratio_function, 'minimize': False},  # Прибыль
    {'function': var_function, 'minimize': True}     # Риск
]

# Шаг 7: Инициализация оптимизатора NSGA-II
optimizer = NSGA_II(
    population_size=500,
    max_generations=200,
    variable_bounds=variable_bounds,
    objectives=objectives,
    mutation_rate=0.5,
    crossover_rate=0.9
)

# Шаг 8: Запуск оптимизации
optimizer.run()

# Шаг 9: Получение и вывод Парето-фронта
optimizer.print_pareto_front()

# Шаг 10: Визуализация Парето-фронта
optimizer.visualize_pareto_front()
