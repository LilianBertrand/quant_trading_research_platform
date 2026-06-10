from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown(equity: pd.Series) -> pd.Series:
    equity = equity.dropna()
    if equity.empty:
        return equity
    return equity / equity.cummax() - 1


def risk_metrics(returns: pd.Series, equity: pd.Series | None = None, benchmark_returns: pd.Series | None = None, risk_free_rate: float = 0.0) -> dict:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {}
    equity = equity.dropna() if equity is not None else (1 + r).cumprod()
    years = max(len(r) / TRADING_DAYS, 1 / TRADING_DAYS)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 1 and equity.iloc[0] != 0 else (1 + r).prod() - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    excess = r - risk_free_rate / TRADING_DAYS
    sharpe = excess.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() != 0 else np.nan
    downside = r[r < 0].std()
    sortino = excess.mean() / downside * np.sqrt(TRADING_DAYS) if downside and not np.isnan(downside) else np.nan
    dd = drawdown(equity)
    max_dd = dd.min() if not dd.empty else np.nan
    var95 = r.quantile(0.05)
    cvar95 = r[r <= var95].mean() if len(r[r <= var95]) else np.nan
    metrics = {
        'Total Return': total_return,
        'CAGR': cagr,
        'Annual Volatility': vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': cagr / abs(max_dd) if max_dd and max_dd < 0 else np.nan,
        'Historical VaR 95%': var95,
        'Historical CVaR 95%': cvar95,
        'Best Day': r.max(),
        'Worst Day': r.min(),
        'Win Rate': (r > 0).mean(),
    }
    if benchmark_returns is not None:
        aligned = pd.concat([r.rename('strategy'), benchmark_returns.rename('benchmark')], axis=1).dropna()
        if len(aligned) > 5 and aligned['benchmark'].var() != 0:
            cov = aligned.cov().loc['strategy', 'benchmark']
            beta = cov / aligned['benchmark'].var()
            alpha = (aligned['strategy'].mean() - beta * aligned['benchmark'].mean()) * TRADING_DAYS
            active = aligned['strategy'] - aligned['benchmark']
            tracking_error = active.std() * np.sqrt(TRADING_DAYS)
            info_ratio = active.mean() / active.std() * np.sqrt(TRADING_DAYS) if active.std() != 0 else np.nan
            metrics.update({'Beta': beta, 'Annualized Alpha': alpha, 'Tracking Error': tracking_error, 'Information Ratio': info_ratio})
    return metrics


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    monthly = (1 + returns.dropna()).resample('ME').prod() - 1
    table = monthly.to_frame('Return')
    table['Year'] = table.index.year
    table['Month'] = table.index.strftime('%b')
    return table.pivot(index='Year', columns='Month', values='Return')


def rolling_metrics(returns: pd.Series, benchmark_returns: pd.Series | None = None, window: int = 63) -> pd.DataFrame:
    r = returns.dropna()
    out = pd.DataFrame(index=r.index)
    out['Rolling Volatility'] = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    out['Rolling Sharpe'] = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    out['Rolling Max Drawdown'] = r.rolling(window).apply(lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min(), raw=False)
    if benchmark_returns is not None:
        aligned = pd.concat([r.rename('strategy'), benchmark_returns.rename('benchmark')], axis=1).dropna()
        beta = aligned['strategy'].rolling(window).cov(aligned['benchmark']) / aligned['benchmark'].rolling(window).var()
        out['Rolling Beta'] = beta.reindex(out.index)
    return out.replace([np.inf, -np.inf], np.nan)


def stress_tests(returns: pd.Series) -> pd.DataFrame:
    r = returns.dropna()
    scenarios = {
        'One-Day -5% Shock': -0.05,
        'One-Day -10% Shock': -0.10,
        'Volatility Spike x2': r.mean() - 2 * r.std(),
        'Worst Historical Day': r.min(),
        'Worst Historical Week': r.rolling(5).sum().min(),
        'Worst Historical Month': r.rolling(21).sum().min(),
    }
    return pd.DataFrame({'Scenario Return': scenarios}).sort_values('Scenario Return')
