# Quant Trading Research Platform

Professional Python and Streamlit platform for quantitative trading research, strategy backtesting, portfolio allocation, risk analytics, walk-forward testing, and market regime detection.

This version is designed to be stable on macOS / VS Code environments. The dashboard works without `scikit-learn` or `openpyxl`, while still supporting a Random Forest regime classifier when `scikit-learn` is installed.

## Features

### Strategy Research
- Momentum strategy
- Mean reversion strategy
- Moving average crossover strategy
- Pairs trading strategy
- Vectorized backtesting engine
- Gross vs net performance after transaction costs and slippage
- Trade log and turnover analytics

### Risk Analytics
- Sharpe ratio
- Sortino ratio
- CAGR
- Maximum drawdown
- Calmar ratio
- Historical VaR 95%
- Historical CVaR 95%
- Rolling volatility
- Rolling Sharpe
- Rolling beta
- Stress scenario analysis

### Portfolio Backtesting
- Equal-weight allocation
- Inverse-volatility allocation / risk parity proxy
- Minimum-variance allocation
- Monthly or quarterly rebalancing
- Contribution to risk
- Benchmark-relative metrics

### Robustness and Walk-Forward Analysis
- Parameter grid search
- Parameter robustness heatmap
- In-sample vs out-of-sample Sharpe
- Rolling walk-forward windows

### Market Regime Detection
- Random Forest classifier if `scikit-learn` is available
- Rule-based fallback detector if `scikit-learn` is unavailable
- Bull / Bear / Range / High Volatility regimes
- Technical feature engineering
- Feature importance analysis
- Regime-adjusted strategy comparison
- Clean accuracy display: shows `N/A` instead of `nan%` when accuracy is not applicable

### Reporting
- Excel-compatible `.xlsx` strategy report export
- No `openpyxl` dependency required
- Metrics, equity curve, daily returns, trade log, and extra research sheets

## Tech Stack

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- yfinance
- Optional: scikit-learn for Random Forest regime classification


## Live Demo

https://quanttradingresearchplatform-yprk9q88wmpv9hbf4ab4yq.streamlit.app/

