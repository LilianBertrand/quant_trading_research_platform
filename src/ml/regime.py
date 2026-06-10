from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = None
    accuracy_score = None
    classification_report = None
    SKLEARN_AVAILABLE = False


def make_features(prices: pd.Series) -> pd.DataFrame:
    s = prices.astype(float).dropna()
    rets = s.pct_change()
    features = pd.DataFrame(index=s.index)
    features['ret_5d'] = s.pct_change(5)
    features['ret_20d'] = s.pct_change(20)
    features['vol_20d'] = rets.rolling(20).std() * np.sqrt(252)
    features['vol_60d'] = rets.rolling(60).std() * np.sqrt(252)
    features['ma_ratio_20_100'] = s.rolling(20).mean() / s.rolling(100).mean() - 1
    features['drawdown'] = s / s.cummax() - 1
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    features['rsi_14'] = 100 - 100 / (1 + rs)
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def _label_regimes(features: pd.DataFrame) -> pd.Series:
    labels = pd.Series('Range', index=features.index, dtype='object')
    if features.empty:
        return labels.rename('Regime')

    high_vol_threshold = features['vol_20d'].quantile(0.75)
    bull = (
        (features['ret_20d'] > 0.02)
        & (features['ma_ratio_20_100'] > 0)
        & (features['drawdown'] > -0.12)
    )
    bear = (
        (features['ret_20d'] < -0.02)
        | (features['ma_ratio_20_100'] < -0.025)
        | (features['drawdown'] < -0.18)
    )
    high_vol = features['vol_20d'] > high_vol_threshold

    labels[bull] = 'Bull'
    labels[bear] = 'Bear'
    labels[high_vol] = 'High Volatility'
    return labels.rename('Regime')


def _rule_based_result(features: pd.DataFrame, reason: str = '') -> dict:
    regimes = _label_regimes(features)
    counts = regimes.value_counts().to_dict()
    total = max(len(regimes), 1)
    dominant = regimes.value_counts().idxmax() if len(regimes) else 'N/A'

    importance = pd.DataFrame(
        {
            'Feature': ['ret_20d', 'ma_ratio_20_100', 'vol_20d', 'drawdown', 'rsi_14', 'ret_5d', 'vol_60d'],
            'Importance': [0.24, 0.22, 0.20, 0.16, 0.08, 0.06, 0.04],
        }
    )

    report_lines = [
        'Rule-based market regime detector active.',
        'The dashboard remains fully functional without scikit-learn.',
    ]
    if reason:
        report_lines.append(f'Reason: {reason}')
    report_lines += [
        f'Observations classified: {len(regimes)}',
        f'Dominant regime: {dominant}',
        '',
        'Regime distribution:',
    ]
    for name, count in counts.items():
        report_lines.append(f'- {name}: {count} observations ({count / total:.1%})')

    return {
        'model_name': 'Rule-based fallback',
        'uses_sklearn': False,
        'features': features,
        'regimes': regimes,
        'test_predictions': regimes.iloc[int(len(regimes) * 0.75):].rename('Predicted Regime'),
        'accuracy': None,
        'feature_importance': importance,
        'classification_report': '\n'.join(report_lines),
    }


def train_regime_classifier(prices: pd.Series) -> dict:
    """Detect market regimes with Random Forest when available, otherwise a robust fallback.

    The function is intentionally defensive: the Streamlit dashboard must not crash
    if scikit-learn is missing, incompatible with the local Python version, or unable
    to train on the selected dataset.
    """
    features = make_features(prices)
    if len(features) < 120:
        raise ValueError('not enough observations for regime detection; use a longer date range')

    labels = _label_regimes(features)
    if not SKLEARN_AVAILABLE:
        return _rule_based_result(features, reason='scikit-learn is not installed in this Python environment')

    try:
        split = int(len(features) * 0.75)
        if split < 80 or len(features) - split < 20:
            return _rule_based_result(features, reason='not enough train/test observations for Random Forest')

        X_train = features.iloc[:split]
        y_train = labels.iloc[:split]
        X_test = features.iloc[split:]
        y_test = labels.iloc[split:]

        if y_train.nunique() < 2:
            return _rule_based_result(features, reason='training sample contains fewer than two regimes')

        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced_subsample',
        )
        model.fit(X_train, y_train)
        test_pred = pd.Series(model.predict(X_test), index=X_test.index, name='Predicted Regime')
        all_pred = pd.Series(model.predict(features), index=features.index, name='Regime')
        accuracy = float(accuracy_score(y_test, test_pred))

        importance = pd.DataFrame(
            {'Feature': features.columns, 'Importance': model.feature_importances_}
        ).sort_values('Importance', ascending=False)

        report = classification_report(y_test, test_pred, zero_division=0)
        report = (
            'Random Forest market regime classifier active.\n'
            f'Train observations: {len(X_train)}\n'
            f'Test observations: {len(X_test)}\n'
            f'Out-of-sample accuracy: {accuracy:.2%}\n\n'
            + report
        )

        return {
            'model_name': 'Random Forest classifier',
            'uses_sklearn': True,
            'model': model,
            'features': features,
            'regimes': all_pred.rename('Regime'),
            'test_predictions': test_pred,
            'accuracy': accuracy,
            'feature_importance': importance,
            'classification_report': report,
        }
    except Exception as exc:
        return _rule_based_result(features, reason=f'Random Forest fallback triggered: {exc}')


def regime_adjusted_signal(base_signal: pd.Series, regimes: pd.Series) -> pd.Series:
    aligned = regimes.reindex(base_signal.index).ffill()
    adjusted = base_signal.copy().astype(float)
    adjusted[aligned == 'Bear'] = adjusted[aligned == 'Bear'].clip(upper=0)
    adjusted[aligned == 'High Volatility'] = adjusted[aligned == 'High Volatility'] * 0.5
    adjusted[aligned == 'Bull'] = adjusted[aligned == 'Bull'].clip(lower=0)
    return adjusted.fillna(0.0).clip(-1, 1)
