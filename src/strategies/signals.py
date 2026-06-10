from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_signal(prices: pd.Series, lookback: int = 60, threshold: float = 0.0) -> pd.Series:
    score = prices.pct_change(lookback)
    signal = pd.Series(0.0, index=prices.index)
    signal[score > threshold] = 1.0
    signal[score < -threshold] = -1.0
    return signal.fillna(0.0)


def mean_reversion_signal(prices: pd.Series, window: int = 20, z_entry: float = 1.25, z_exit: float = 0.25) -> pd.Series:
    ma = prices.rolling(window).mean()
    vol = prices.rolling(window).std().replace(0, np.nan)
    z = (prices - ma) / vol
    signal = pd.Series(0.0, index=prices.index)
    signal[z < -z_entry] = 1.0
    signal[z > z_entry] = -1.0
    signal[z.abs() < z_exit] = 0.0
    return signal.ffill().fillna(0.0).clip(-1, 1)


def moving_average_crossover_signal(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    if fast >= slow:
        fast = max(2, slow // 4)
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signal = pd.Series(0.0, index=prices.index)
    signal[fast_ma > slow_ma] = 1.0
    signal[fast_ma < slow_ma] = -1.0
    return signal.fillna(0.0)


def pairs_trading_signal(a: pd.Series, b: pd.Series, window: int = 60, z_entry: float = 1.5, z_exit: float = 0.3) -> pd.DataFrame:
    data = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    beta = (data['a'].rolling(window).cov(data['b']) / data['b'].rolling(window).var()).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    spread = data['a'] - beta * data['b']
    z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std().replace(0, np.nan)
    pos_a = pd.Series(0.0, index=data.index)
    pos_a[z > z_entry] = -1.0
    pos_a[z < -z_entry] = 1.0
    pos_a[z.abs() < z_exit] = 0.0
    pos_a = pos_a.ffill().fillna(0.0)
    pos_b = -pos_a * beta
    positions = pd.concat([pos_a, pos_b], axis=1)
    positions.columns = [a.name or 'Asset A', b.name or 'Asset B']
    gross_abs = positions.abs().sum(axis=1).replace(0, np.nan)
    positions = positions.div(gross_abs, axis=0).fillna(0.0)
    return positions
