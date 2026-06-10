from __future__ import annotations

import numpy as np
import pandas as pd


def _cost_rate(fee_bps: float, slippage_bps: float) -> float:
    return max(0.0, float(fee_bps) + float(slippage_bps)) / 10_000


def build_trade_log(prices: pd.Series, position: pd.Series) -> pd.DataFrame:
    prices = prices.reindex(position.index).ffill()
    change = position.diff().fillna(position)
    events = change[change.abs() > 1e-12]
    rows = []
    entry_date = entry_price = side = None
    for date in events.index:
        pos = float(position.loc[date])
        price = float(prices.loc[date])
        if side is None and pos != 0:
            side = 'Long' if pos > 0 else 'Short'
            entry_date = date
            entry_price = price
        elif side is not None and pos == 0:
            pnl = price / entry_price - 1 if side == 'Long' else entry_price / price - 1
            rows.append({
                'Entry Date': entry_date,
                'Exit Date': date,
                'Side': side,
                'Entry Price': entry_price,
                'Exit Price': price,
                'Trade Return': pnl,
                'Holding Days': (pd.Timestamp(date) - pd.Timestamp(entry_date)).days,
            })
            entry_date = entry_price = side = None
        elif side is not None and pos != 0:
            new_side = 'Long' if pos > 0 else 'Short'
            if new_side != side:
                pnl = price / entry_price - 1 if side == 'Long' else entry_price / price - 1
                rows.append({
                    'Entry Date': entry_date,
                    'Exit Date': date,
                    'Side': side,
                    'Entry Price': entry_price,
                    'Exit Price': price,
                    'Trade Return': pnl,
                    'Holding Days': (pd.Timestamp(date) - pd.Timestamp(entry_date)).days,
                })
                side = new_side
                entry_date = date
                entry_price = price
    return pd.DataFrame(rows)


def backtest_single_asset(prices: pd.Series, signal: pd.Series, initial_capital=100_000, fee_bps=2.0, slippage_bps=1.0) -> dict:
    prices = prices.dropna().astype(float)
    signal = signal.reindex(prices.index).ffill().fillna(0.0).clip(-1, 1)
    asset_returns = prices.pct_change().fillna(0.0)
    position = signal.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    gross_returns = position * asset_returns
    costs = turnover * _cost_rate(fee_bps, slippage_bps)
    net_returns = gross_returns - costs
    equity = initial_capital * (1 + net_returns).cumprod()
    gross_equity = initial_capital * (1 + gross_returns).cumprod()
    return {
        'returns': net_returns,
        'gross_returns': gross_returns,
        'equity': equity,
        'gross_equity': gross_equity,
        'position': position,
        'turnover': turnover,
        'costs': costs,
        'trades': build_trade_log(prices, position),
    }


def backtest_multi_asset(prices: pd.DataFrame, weights: pd.DataFrame, initial_capital=100_000, fee_bps=2.0, slippage_bps=1.0) -> dict:
    prices = prices.dropna(how='all').ffill().astype(float)
    returns = prices.pct_change().fillna(0.0)
    weights = weights.reindex(prices.index).ffill().fillna(0.0)
    weights = weights.reindex(columns=prices.columns).fillna(0.0)
    gross_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * _cost_rate(fee_bps, slippage_bps)
    net_returns = gross_returns - costs
    equity = initial_capital * (1 + net_returns).cumprod()
    gross_equity = initial_capital * (1 + gross_returns).cumprod()
    return {
        'returns': net_returns,
        'gross_returns': gross_returns,
        'equity': equity,
        'gross_equity': gross_equity,
        'weights': weights,
        'turnover': turnover,
        'costs': costs,
    }
