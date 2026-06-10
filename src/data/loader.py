from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_tickers(tickers: str | list[str]) -> list[str]:
    if isinstance(tickers, str):
        raw = tickers.replace(';', ',').replace('\n', ',').split(',')
    else:
        raw = tickers
    clean = []
    for item in raw:
        value = str(item).strip().upper()
        if value and value not in clean:
            clean.append(value)
    return clean


def generate_demo_prices(tickers: str | list[str], start='2018-01-01', end=None, seed: int = 42) -> pd.DataFrame:
    tickers = normalize_tickers(tickers) or ['SPY', 'QQQ', 'TLT', 'GLD']
    end = pd.Timestamp(end or pd.Timestamp.today()).normalize()
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) < 60:
        dates = pd.bdate_range(end=end, periods=252)
    rng = np.random.default_rng(seed)
    market = rng.normal(0.00025, 0.0095, len(dates))
    prices = pd.DataFrame(index=dates)
    for i, ticker in enumerate(tickers):
        beta = 0.55 + 0.12 * (i % 5)
        idio = rng.normal(0.00008 + 0.00003 * i, 0.010 + 0.002 * (i % 4), len(dates))
        cyclical = 0.0008 * np.sin(np.linspace(0, 14 * np.pi, len(dates)) + i)
        returns = beta * market + (1 - min(beta, 0.9)) * idio + cyclical / 20
        prices[ticker] = 100 * np.exp(np.cumsum(returns))
    return prices.ffill().dropna()


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        first_level = raw.columns.get_level_values(0)
        second_level = raw.columns.get_level_values(1)
        if 'Close' in first_level:
            prices = raw['Close']
        elif 'Close' in second_level:
            prices = raw.xs('Close', axis=1, level=1)
        else:
            prices = raw.iloc[:, :len(tickers)]
    else:
        close_col = 'Close' if 'Close' in raw.columns else raw.columns[0]
        prices = raw[[close_col]].rename(columns={close_col: tickers[0] if tickers else 'ASSET'})
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0] if tickers else 'ASSET')
    prices.columns = [str(c).upper() for c in prices.columns]
    return prices


def load_prices(tickers: str | list[str], start='2018-01-01', end=None, use_demo=False) -> pd.DataFrame:
    tickers = normalize_tickers(tickers)
    if not tickers:
        tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
    if use_demo:
        return generate_demo_prices(tickers, start, end)
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
        prices = _extract_close(raw, tickers)
        prices = prices.replace([np.inf, -np.inf], np.nan).ffill().dropna(axis=0, how='all').dropna(axis=1, how='all')
        if prices.empty or prices.shape[1] == 0:
            raise ValueError('empty downloaded data')
        return prices
    except Exception:
        return generate_demo_prices(tickers, start, end)


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='all')
