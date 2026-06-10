from __future__ import annotations

import pandas as pd

from src.backtesting.engine import backtest_single_asset
from src.risk.metrics import risk_metrics
from src.strategies.signals import momentum_signal, mean_reversion_signal, moving_average_crossover_signal


def _signal(strategy: str, prices: pd.Series, parameter: int | float) -> pd.Series:
    if strategy == 'Momentum':
        return momentum_signal(prices, lookback=int(parameter))
    if strategy == 'Mean Reversion':
        return mean_reversion_signal(prices, window=int(parameter), z_entry=1.25)
    return moving_average_crossover_signal(prices, fast=int(parameter), slow=max(int(parameter) * 4, int(parameter) + 20))


def parameter_grid_search(prices: pd.Series, strategy: str, fee_bps=2.0, slippage_bps=1.0) -> pd.DataFrame:
    if strategy == 'Momentum':
        params = [20, 40, 60, 90, 120, 180, 240]
    elif strategy == 'Mean Reversion':
        params = [10, 15, 20, 30, 45, 60, 90]
    else:
        params = [10, 20, 30, 50, 75]
    rows = []
    for p in params:
        bt = backtest_single_asset(prices, _signal(strategy, prices, p), fee_bps=fee_bps, slippage_bps=slippage_bps)
        m = risk_metrics(bt['returns'], bt['equity'])
        rows.append({'Strategy': strategy, 'Parameter': p, **m})
    return pd.DataFrame(rows)


def walk_forward_analysis(prices: pd.Series, strategy: str, train_days=504, test_days=126, fee_bps=2.0, slippage_bps=1.0) -> pd.DataFrame:
    prices = prices.dropna()
    if len(prices) < train_days + test_days:
        return pd.DataFrame()
    rows = []
    start = 0
    while start + train_days + test_days <= len(prices):
        train = prices.iloc[start:start + train_days]
        test = prices.iloc[start + train_days:start + train_days + test_days]
        grid = parameter_grid_search(train, strategy, fee_bps, slippage_bps)
        best = grid.sort_values('Sharpe Ratio', ascending=False).iloc[0]
        signal = _signal(strategy, test, best['Parameter'])
        bt = backtest_single_asset(test, signal, fee_bps=fee_bps, slippage_bps=slippage_bps)
        out = risk_metrics(bt['returns'], bt['equity'])
        rows.append({
            'Train Start': train.index[0],
            'Train End': train.index[-1],
            'Test Start': test.index[0],
            'Test End': test.index[-1],
            'Best Parameter': best['Parameter'],
            'In-Sample Sharpe': best['Sharpe Ratio'],
            'Out-of-Sample Sharpe': out.get('Sharpe Ratio'),
            'Out-of-Sample Return': out.get('Total Return'),
            'Out-of-Sample Max Drawdown': out.get('Max Drawdown'),
        })
        start += test_days
    return pd.DataFrame(rows)
