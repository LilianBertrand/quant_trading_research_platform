from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.backtesting.engine import backtest_multi_asset, backtest_single_asset
from src.data.loader import daily_returns, load_prices
from src.ml.regime import regime_adjusted_signal, train_regime_classifier
from src.portfolio.allocation import contribution_to_risk, equal_weight, inverse_volatility, minimum_variance
from src.portfolio.walk_forward import parameter_grid_search, walk_forward_analysis
from src.reporting.exporter import excel_report
from src.risk.metrics import drawdown, monthly_returns, risk_metrics, rolling_metrics, stress_tests
from src.strategies.signals import mean_reversion_signal, momentum_signal, moving_average_crossover_signal, pairs_trading_signal

st.set_page_config(page_title='Quant Trading Research Platform', page_icon='📈', layout='wide')

st.markdown(
    '''
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {background: rgba(49, 51, 63, 0.35); border: 1px solid rgba(250,250,250,0.08); padding: 14px; border-radius: 14px;}
    .small-note {font-size: 0.88rem; color: #9ca3af;}
    </style>
    ''',
    unsafe_allow_html=True,
)

st.title('Quant Trading Research Platform')
st.caption('Backtesting, portfolio construction, risk analytics, walk-forward research and ML market regime detection.')

with st.sidebar:
    st.header('Research Setup')
    tickers = st.text_area('Universe', value='SPY, QQQ, TLT, GLD', help='Use commas or one ticker per line.')
    start = st.date_input('Start date', value=pd.Timestamp('2018-01-01'))
    end = st.date_input('End date', value=pd.Timestamp.today())
    use_demo = st.toggle('Use demo data / fallback mode', value=False)
    initial_capital = st.number_input('Initial capital', min_value=1_000, value=100_000, step=10_000)
    fee_bps = st.slider('Transaction fees, bps', 0.0, 50.0, 2.0, 0.5)
    slippage_bps = st.slider('Slippage, bps', 0.0, 50.0, 1.0, 0.5)
    benchmark_ticker = st.text_input('Benchmark ticker', value='SPY').upper().strip()
    page = st.radio(
        'Module',
        [
            'Executive Overview',
            'Strategy Lab',
            'Strategy Comparison',
            'Portfolio Backtesting',
            'Risk Management',
            'Walk-Forward Analysis',
            'ML Regime Detection',
            'Report Export',
        ],
    )

prices = load_prices(tickers, start=str(start), end=str(end), use_demo=use_demo)
if prices.empty:
    st.error('No data available. Enable demo data/fallback mode or verify your tickers.')
    st.stop()

returns = daily_returns(prices)
asset = st.sidebar.selectbox('Primary asset', prices.columns.tolist())
benchmark_returns = returns[benchmark_ticker] if benchmark_ticker in returns.columns else returns.iloc[:, 0]


def fmt_metric(key: str, val):
    if val is None or pd.isna(val):
        return 'n/a'
    if key in {'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Beta', 'Information Ratio'}:
        return f'{val:.2f}'
    if key in {'Annualized Alpha', 'Tracking Error', 'Total Return', 'CAGR', 'Annual Volatility', 'Max Drawdown', 'Historical VaR 95%', 'Historical CVaR 95%', 'Best Day', 'Worst Day', 'Win Rate'}:
        return f'{val:.2%}'
    return f'{val:.4f}' if isinstance(val, float) else str(val)


def metric_grid(metrics: dict):
    keys = ['Total Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Historical VaR 95%']
    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        col.metric(key, fmt_metric(key, metrics.get(key)))


def plot_equity(bt: dict, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt['equity'].index, y=bt['equity'], mode='lines', name='Net Equity'))
    fig.add_trace(go.Scatter(x=bt['gross_equity'].index, y=bt['gross_equity'], mode='lines', name='Gross Equity', line={'dash': 'dot'}))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Portfolio Value', hovermode='x unified')
    st.plotly_chart(fig, width='stretch')


def strategy_backtest(strategy: str, asset_name: str, with_sidebar_params: bool = True):
    s = prices[asset_name]
    if strategy == 'Momentum':
        lookback = st.sidebar.slider('Momentum lookback', 20, 240, 60, 10) if with_sidebar_params else 60
        signal = momentum_signal(s, lookback=lookback)
        return backtest_single_asset(s, signal, initial_capital, fee_bps, slippage_bps)
    if strategy == 'Mean Reversion':
        window = st.sidebar.slider('Mean reversion window', 10, 120, 20, 5) if with_sidebar_params else 20
        z_entry = st.sidebar.slider('Z-score entry', 0.5, 3.0, 1.25, 0.25) if with_sidebar_params else 1.25
        signal = mean_reversion_signal(s, window=window, z_entry=z_entry)
        return backtest_single_asset(s, signal, initial_capital, fee_bps, slippage_bps)
    if strategy == 'Moving Average Crossover':
        fast = st.sidebar.slider('Fast moving average', 10, 100, 50, 5) if with_sidebar_params else 50
        slow = st.sidebar.slider('Slow moving average', 80, 300, 200, 10) if with_sidebar_params else 200
        signal = moving_average_crossover_signal(s, fast=fast, slow=slow)
        return backtest_single_asset(s, signal, initial_capital, fee_bps, slippage_bps)
    pair_candidates = [c for c in prices.columns if c != asset_name]
    if not pair_candidates:
        st.warning('Pairs trading requires at least two assets.')
        return None
    second = st.sidebar.selectbox('Second asset for pair', pair_candidates) if with_sidebar_params else pair_candidates[0]
    window = st.sidebar.slider('Pairs window', 20, 180, 60, 10) if with_sidebar_params else 60
    z_entry = st.sidebar.slider('Pairs z-score entry', 0.5, 3.0, 1.5, 0.25) if with_sidebar_params else 1.5
    positions = pairs_trading_signal(prices[asset_name], prices[second], window=window, z_entry=z_entry)
    return backtest_multi_asset(prices[[asset_name, second]], positions, initial_capital, fee_bps, slippage_bps)


if page == 'Executive Overview':
    st.subheader('Platform Overview')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Assets Loaded', len(prices.columns))
    c2.metric('Observations', len(prices))
    c3.metric('Start', str(prices.index.min().date()))
    c4.metric('End', str(prices.index.max().date()))
    normalized = prices / prices.iloc[0] * 100
    st.plotly_chart(px.line(normalized, title='Normalized Asset Prices'), width='stretch')
    st.subheader('Daily Return Correlation')
    st.plotly_chart(px.imshow(returns.corr(), text_auto='.2f', title='Correlation Heatmap'), width='stretch')
    st.subheader('Data Preview')
    st.dataframe(prices.tail(20), width='stretch')

elif page == 'Strategy Lab':
    strategy = st.sidebar.selectbox('Strategy', ['Momentum', 'Mean Reversion', 'Moving Average Crossover', 'Pairs Trading'])
    bt = strategy_backtest(strategy, asset)
    if bt is None:
        st.stop()
    metrics = risk_metrics(bt['returns'], bt['equity'], benchmark_returns)
    metric_grid(metrics)
    c1, c2 = st.columns([2, 1])
    with c1:
        plot_equity(bt, f'{strategy} on {asset}')
    with c2:
        st.plotly_chart(px.area(drawdown(bt['equity']), title='Drawdown'), width='stretch')
    st.subheader('Transaction Costs and Turnover')
    cost_frame = pd.DataFrame({'Cumulative Gross Return': bt['gross_returns'].cumsum(), 'Cumulative Net Return': bt['returns'].cumsum(), 'Cumulative Costs': bt['costs'].cumsum()})
    st.plotly_chart(px.line(cost_frame, title='Gross vs Net Return Impact'), width='stretch')
    st.dataframe(pd.DataFrame(metrics.items(), columns=['Metric', 'Value']), width='stretch')
    if bt.get('trades') is not None and not bt.get('trades').empty:
        st.subheader('Trade Log')
        st.dataframe(bt['trades'], width='stretch')

elif page == 'Strategy Comparison':
    rows, curves, dds = [], {}, {}
    for strat in ['Momentum', 'Mean Reversion', 'Moving Average Crossover']:
        bt = strategy_backtest(strat, asset, with_sidebar_params=False)
        m = risk_metrics(bt['returns'], bt['equity'], benchmark_returns)
        rows.append({'Strategy': strat, **m, 'Trades': len(bt.get('trades', [])), 'Turnover': bt['turnover'].mean() * 252})
        curves[strat] = bt['equity']
        dds[strat] = drawdown(bt['equity'])
    if len(prices.columns) >= 2:
        a, b = prices.columns[:2]
        positions = pairs_trading_signal(prices[a], prices[b])
        bt = backtest_multi_asset(prices[[a, b]], positions, initial_capital, fee_bps, slippage_bps)
        rows.append({'Strategy': f'Pairs Trading {a}/{b}', **risk_metrics(bt['returns'], bt['equity'], benchmark_returns), 'Trades': 'n/a', 'Turnover': bt['turnover'].mean() * 252})
        curves['Pairs Trading'] = bt['equity']
        dds['Pairs Trading'] = drawdown(bt['equity'])
    comp = pd.DataFrame(rows).sort_values('Sharpe Ratio', ascending=False)
    st.subheader('Strategy Ranking')
    st.dataframe(comp, width='stretch')
    st.plotly_chart(px.line(pd.DataFrame(curves), title='Equity Curves'), width='stretch')
    st.plotly_chart(px.line(pd.DataFrame(dds), title='Drawdowns'), width='stretch')
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(comp, x='Strategy', y='Sharpe Ratio', title='Sharpe Ratio Ranking'), width='stretch')
    with c2:
        st.plotly_chart(px.bar(comp, x='Strategy', y='Max Drawdown', title='Maximum Drawdown by Strategy'), width='stretch')

elif page == 'Portfolio Backtesting':
    method = st.sidebar.selectbox('Allocation method', ['Equal Weight', 'Risk Parity Proxy', 'Minimum Variance'])
    rebalance = st.sidebar.selectbox('Rebalancing', ['ME', 'QE'], format_func=lambda x: 'Monthly' if x == 'ME' else 'Quarterly')
    if method == 'Equal Weight':
        weights = equal_weight(prices, rebalance=rebalance)
    elif method == 'Risk Parity Proxy':
        weights = inverse_volatility(prices, rebalance=rebalance)
    else:
        weights = minimum_variance(prices, rebalance=rebalance)
    bt = backtest_multi_asset(prices, weights, initial_capital, fee_bps, slippage_bps)
    metrics = risk_metrics(bt['returns'], bt['equity'], benchmark_returns)
    metric_grid(metrics)
    plot_equity(bt, f'{method} Portfolio')
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.area(weights, title='Portfolio Weights Over Time'), width='stretch')
    with c2:
        latest_rc = contribution_to_risk(weights.iloc[-1], returns[weights.columns])
        st.plotly_chart(px.bar(latest_rc, title='Latest Risk Contribution'), width='stretch')
    st.subheader('Latest Portfolio Weights')
    st.dataframe(weights.tail(1).T.rename(columns={weights.index[-1]: 'Weight'}), width='stretch')

elif page == 'Risk Management':
    strategy = st.sidebar.selectbox('Risk analysis strategy', ['Momentum', 'Mean Reversion', 'Moving Average Crossover'])
    bt = strategy_backtest(strategy, asset, with_sidebar_params=False)
    metrics = risk_metrics(bt['returns'], bt['equity'], benchmark_returns)
    metric_grid(metrics)
    st.subheader('Rolling Risk Metrics')
    roll = rolling_metrics(bt['returns'], benchmark_returns)
    st.plotly_chart(px.line(roll, title='Rolling Volatility, Sharpe, Drawdown and Beta'), width='stretch')
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(bt['returns'], nbins=80, title='Daily Returns Distribution'), width='stretch')
    with c2:
        try:
            heat = monthly_returns(bt['returns'])
            st.plotly_chart(px.imshow(heat, text_auto='.1%', title='Monthly Returns Heatmap'), width='stretch')
        except Exception:
            st.info('Monthly heatmap unavailable for this sample.')
    st.subheader('Stress Scenarios')
    st.dataframe(stress_tests(bt['returns']), width='stretch')
    st.subheader('Worst 10 Days')
    st.dataframe(bt['returns'].nsmallest(10).rename('Daily Return').to_frame(), width='stretch')

elif page == 'Walk-Forward Analysis':
    strategy = st.sidebar.selectbox('Strategy for walk-forward', ['Momentum', 'Mean Reversion', 'Moving Average Crossover'])
    grid = parameter_grid_search(prices[asset], strategy, fee_bps, slippage_bps)
    st.subheader('Parameter Robustness')
    st.dataframe(grid, width='stretch')
    heat = grid[['Parameter', 'Sharpe Ratio', 'CAGR', 'Max Drawdown']].set_index('Parameter').T
    st.plotly_chart(px.imshow(heat, text_auto='.2f', title='Parameter Robustness Heatmap'), width='stretch')
    wf = walk_forward_analysis(prices[asset], strategy, fee_bps=fee_bps, slippage_bps=slippage_bps)
    st.subheader('Walk-Forward Out-of-Sample Results')
    if wf.empty:
        st.warning('Not enough data for walk-forward analysis. Use a longer date range.')
    else:
        st.dataframe(wf, width='stretch')
        st.plotly_chart(px.line(wf, x='Test End', y=['In-Sample Sharpe', 'Out-of-Sample Sharpe'], title='In-Sample vs Out-of-Sample Sharpe'), width='stretch')

elif page == 'ML Regime Detection':
    st.info(
        'This module uses a Random Forest classifier when scikit-learn is available. '
        'If scikit-learn is missing or incompatible, the dashboard automatically uses '
        'a rule-based market regime detector so the app never crashes.'
    )
    try:
        result = train_regime_classifier(prices[asset])
        model_name = result.get('model_name', 'Market regime detector')
        accuracy = result.get('accuracy')

        c0, c00 = st.columns(2)
        c0.metric('Model engine', model_name)
        if accuracy is None or pd.isna(accuracy):
            c00.metric('Out-of-sample accuracy', 'N/A')
            st.caption('Accuracy is only shown when the Random Forest engine is active and a valid test set exists.')
        else:
            c00.metric('Out-of-sample accuracy', f'{accuracy:.2%}')

        c1, c2 = st.columns([2, 1])
        with c1:
            regime_df = pd.concat([prices[asset].rename(asset), result['regimes']], axis=1).dropna()
            st.plotly_chart(px.scatter(regime_df, x=regime_df.index, y=asset, color='Regime', title='Detected Market Regimes'), width='stretch')
        with c2:
            st.plotly_chart(px.bar(result['feature_importance'], x='Feature', y='Importance', title='Feature Importance'), width='stretch')
        base_signal = momentum_signal(prices[asset], 60)
        adjusted_signal = regime_adjusted_signal(base_signal, result['regimes'])
        bt_base = backtest_single_asset(prices[asset], base_signal, initial_capital, fee_bps, slippage_bps)
        bt_adjusted = backtest_single_asset(prices[asset], adjusted_signal, initial_capital, fee_bps, slippage_bps)
        comparison = pd.DataFrame({'Base Momentum': bt_base['equity'], 'Regime-Adjusted Momentum': bt_adjusted['equity']}).dropna()
        st.plotly_chart(px.line(comparison, title='Base Momentum vs Regime-Adjusted Momentum'), width='stretch')
        st.subheader('Model Diagnostics')
        st.text(result['classification_report'])
        st.subheader('Strategy Metrics')
        st.dataframe(pd.DataFrame([risk_metrics(bt_base['returns'], bt_base['equity']), risk_metrics(bt_adjusted['returns'], bt_adjusted['equity'])], index=['Base Momentum', 'Regime Adjusted']), width='stretch')
    except Exception as exc:
        st.warning(f'ML module could not run on this dataset: {exc}')

elif page == 'Report Export':
    strategy = st.sidebar.selectbox('Strategy to export', ['Momentum', 'Mean Reversion', 'Moving Average Crossover'])
    bt = strategy_backtest(strategy, asset, with_sidebar_params=False)
    metrics = risk_metrics(bt['returns'], bt['equity'], benchmark_returns)
    metric_grid(metrics)
    plot_equity(bt, f'Export Preview - {strategy}')
    extra = {
        'Drawdown': drawdown(bt['equity']).rename('Drawdown').to_frame(),
        'Rolling Metrics': rolling_metrics(bt['returns'], benchmark_returns),
        'Stress Tests': stress_tests(bt['returns']),
    }
    report = excel_report(metrics, bt['returns'], bt['equity'], bt.get('trades'), extra)
    st.download_button(
        label='Download Excel Strategy Report',
        data=report,
        file_name='quant_strategy_report.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

with st.expander('Loaded price data'):
    st.dataframe(prices.tail(30), width='stretch')
