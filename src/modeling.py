"""Model fitting and evaluation helpers shared by the modeling notebooks (Weeks 4 to 8).

Every model is wrapped the same way: the preprocessor from
``src.preprocessing`` followed by an estimator, trained to predict
log(ClosePrice) and converted back to dollars when predicting.
"""
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src import preprocessing as prep


def price_model(estimator, log_target=True):
    """Preprocessor + estimator. With ``log_target`` the estimator learns log(price)
    and predictions are exponentiated back to dollars."""
    pipeline = Pipeline([('pre', prep.build_preprocessor()), ('model', estimator)])
    if log_target:
        return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)
    return TransformedTargetRegressor(regressor=pipeline)


def fit_and_score(clean_df, eval_month, n_train_months, make_estimator, log_target=True):
    """Train on the ``n_train_months`` months before ``eval_month`` and score on ``eval_month``.

    Returns ``(model, train, evaluation, metrics)``, where ``metrics`` holds the
    evaluation-month R2, MAPE, MdAPE and MAE plus row counts.
    """
    train, evaluation, _ = prep.make_split(clean_df, eval_month, n_train_months)
    model = price_model(make_estimator(), log_target=log_target)
    model.fit(train[prep.FEATURES], train[prep.TARGET])
    metrics = prep.regression_metrics(evaluation[prep.TARGET], model.predict(evaluation[prep.FEATURES]))
    metrics = {'eval_month': eval_month,
               'train_months': f"{train['CloseMonth'].min()} to {train['CloseMonth'].max()}",
               'train_rows': len(train), 'eval_rows': len(evaluation), **metrics}
    return model, train, evaluation, metrics


def error_by_group(y_true, y_pred, groups):
    """Error summary per group: count, MdAPE, MAPE and the median signed percentage
    error (positive = the model predicts too high)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    signed = 100 * (y_pred - y_true) / y_true
    df = pd.DataFrame({'group': np.asarray(groups), 'signed': signed, 'abs': np.abs(signed)})
    return (df.groupby('group', observed=True)
              .agg(homes=('abs', 'size'), MdAPE=('abs', 'median'), MAPE=('abs', 'mean'),
                   median_signed_error=('signed', 'median')))


def bootstrap_mdape_difference(y_true, pred_a, pred_b, n_resamples=2000, seed=prep.RANDOM_STATE):
    """Paired bootstrap of MdAPE(a) - MdAPE(b) over the same homes.

    Homes are resampled with replacement; for each resample the difference in
    MdAPE between the two models is recomputed. Returns the observed
    difference and the 2.5th and 97.5th percentiles of the resampled
    differences (a 95% interval), all in percentage points.
    """
    y_true = np.asarray(y_true, dtype=float)
    ape_a = np.abs(np.asarray(pred_a, dtype=float) - y_true) / y_true
    ape_b = np.abs(np.asarray(pred_b, dtype=float) - y_true) / y_true
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_resamples, len(y_true)))
    diffs = 100 * (np.median(ape_a[idx], axis=1) - np.median(ape_b[idx], axis=1))
    observed = 100 * (np.median(ape_a) - np.median(ape_b))
    return {'difference': float(observed), 'ci_low': float(np.percentile(diffs, 2.5)),
            'ci_high': float(np.percentile(diffs, 97.5))}
