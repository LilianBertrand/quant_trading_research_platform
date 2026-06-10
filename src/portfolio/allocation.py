from __future__ import annotations

import numpy as np
import pandas as pd


def _rebalance_dates(prices: pd.DataFrame, rebalance: str = 'ME') -> pd.DatetimeIndex:
    return prices.resample(rebalance).last().index.intersection(prices.index)


def equal_weight(prices: pd.DataFrame, rebalance: str = 'ME') -> pd.DataFrame:
    w = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for d in _rebalance_dates(prices, rebalance):
        w.loc[d] = 1 / len(prices.columns)
    return w.ffill().fillna(1 / len(prices.columns))


def inverse_volatility(prices: pd.DataFrame, window: int = 63, rebalance: str = 'ME') -> pd.DataFrame:
    returns = prices.pct_change()
    vol = returns.rolling(window).std()
    w = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for d in _rebalance_dates(prices, rebalance):
        inv = (1 / vol.loc[d].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        row = pd.Series(0.0, index=prices.columns)
        if len(inv) > 0 and inv.sum() != 0:
            row.loc[inv.index] = inv / inv.sum()
        else:
            row[:] = 1 / len(row)
        w.loc[d] = row
    return w.ffill().fillna(1 / len(prices.columns))


def minimum_variance(prices: pd.DataFrame, window: int = 126, rebalance: str = 'ME') -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    n = len(prices.columns)
    w = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for d in _rebalance_dates(prices, rebalance):
        hist = returns.loc[:d].tail(window)
        if len(hist) < max(30, n + 5):
            w.loc[d] = np.repeat(1 / n, n)
            continue
        cov = hist.cov().values
        inv_cov = np.linalg.pinv(cov)
        ones = np.ones(n)
        raw = inv_cov @ ones / (ones.T @ inv_cov @ ones)
        raw = np.clip(raw, 0, 1)
        raw = raw / raw.sum() if raw.sum() > 0 else np.repeat(1 / n, n)
        w.loc[d] = raw
    return w.ffill().fillna(1 / n)


def contribution_to_risk(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    weights = weights.reindex(returns.columns).fillna(0.0)
    cov = returns.cov() * 252
    variance = float(weights.values.T @ cov.values @ weights.values)
    if variance <= 0 or np.isnan(variance):
        return pd.Series(0.0, index=weights.index)
    vol = np.sqrt(variance)
    marginal = cov.values @ weights.values / vol
    contribution = weights.values * marginal / vol
    return pd.Series(contribution, index=weights.index).sort_values(ascending=False)
