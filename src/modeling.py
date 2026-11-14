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


def price_model(estimator, log_target=True, features=None, nearby_k=25):
    """Preprocessor + estimator. With ``log_target`` the estimator learns log(price)
    and predictions are exponentiated back to dollars. ``features`` selects the
    input columns (default: the Week 3 set, prep.FEATURES)."""
    if features is not None and prep.NEARBY_FEATURE in features:
        assert log_target, 'NearbyPricePerSqft expects the log-price target'
    pipeline = Pipeline([('pre', prep.build_preprocessor(features, nearby_k=nearby_k)), ('model', estimator)])
    if log_target:
        return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)
    return TransformedTargetRegressor(regressor=pipeline)


def fit_and_score(clean_df, eval_month, n_train_months, make_estimator, log_target=True, features=None,
                  nearby_k=25):
    """Train on the ``n_train_months`` months before ``eval_month`` and score on ``eval_month``.

    Returns ``(model, train, evaluation, metrics)``, where ``metrics`` holds the
    evaluation-month R2, MAPE, MdAPE and MAE plus row counts.
    """
    train, evaluation, _ = prep.make_split(clean_df, eval_month, n_train_months)
    features = prep.FEATURES if features is None else list(features)
    cols = prep.input_columns(features)
    model = price_model(make_estimator(), log_target=log_target, features=features, nearby_k=nearby_k)
    model.fit(train[cols], train[prep.TARGET])
    metrics = prep.regression_metrics(evaluation[prep.TARGET], model.predict(evaluation[cols]))
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


def one_se_select(y_true, preds, complexity, n_resamples=2000, seed=prep.RANDOM_STATE):
    """One-standard-error rule on a validation month.

    ``preds`` maps each candidate setting to its validation predictions, and
    ``complexity`` maps it to a number where smaller means simpler (for example
    the number of tree nodes). The candidate with the lowest MdAPE is the best;
    for every other candidate, the gap to the best and the standard error of
    that gap are estimated with a paired bootstrap over the validation homes.
    Candidates whose gap is at most one standard error are statistical ties,
    and the simplest of them is selected. Returns ``(selected_key, table)``.
    """
    y_true = np.asarray(y_true, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_resamples, len(y_true)))
    medians, mdape = {}, {}
    for key, pred in preds.items():
        ape = np.abs(np.asarray(pred, dtype=float) - y_true) / y_true
        mdape[key] = 100 * np.median(ape)
        medians[key] = 100 * np.median(ape[idx], axis=1)
    best = min(mdape, key=mdape.get)
    rows = []
    for key in preds:
        gap = mdape[key] - mdape[best]
        se = float(np.std(medians[key] - medians[best]))
        rows.append({'setting': key, 'MdAPE': mdape[key], 'gap_to_best': gap, 'se_of_gap': se,
                     'within_1se': gap <= se, 'complexity': complexity[key]})
    table = pd.DataFrame(rows).set_index('setting')
    selected = table[table['within_1se']].sort_values(['complexity', 'MdAPE']).index[0]
    return selected, table


def prepare_matrices(features, train, evaluation, nearby_k=25):
    """Fit the preprocessor on the training rows once and return
    ``(preprocessor, X_train, X_eval, log_y_train)``.

    Fitting a model on these matrices gives exactly the same result as fitting
    ``price_model`` (the preprocessor learns only from training rows and does
    not depend on the model), but lets many model settings reuse one
    preprocessing pass. Predictions on X_eval are in log price; exponentiate
    them to get dollars.
    """
    cols = prep.input_columns(features)
    log_y = np.log(train[prep.TARGET].to_numpy())
    pre = prep.build_preprocessor(features, nearby_k=nearby_k)
    X_train = pre.fit_transform(train[cols], log_y)
    X_eval = pre.transform(evaluation[cols])
    return pre, X_train, X_eval, log_y


def full_metrics(y_true, y_pred):
    """The Week 8 evaluation measures for one set of predictions, on the price scale.

    R2, MAE and RMSE (dollars), MAPE and MdAPE (%), the share of homes predicted
    within 5%, 10% and 20% of the sale price (%), and the median signed
    percentage error (positive = predicted too high).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    signed = (y_pred - y_true) / y_true
    ape = np.abs(signed)
    return {
        'homes': len(y_true),
        **prep.regression_metrics(y_true, y_pred),
        'RMSE': float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        'within 5%': 100 * float(np.mean(ape <= 0.05)),
        'within 10%': 100 * float(np.mean(ape <= 0.10)),
        'within 20%': 100 * float(np.mean(ape <= 0.20)),
        'median signed error': 100 * float(np.median(signed)),
    }


def segment_metrics(y_true, y_pred, groups):
    """Per-group homes, MAPE, MdAPE, share within 10% and median signed error (all %)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    signed = 100 * (y_pred - y_true) / y_true
    df = pd.DataFrame({'group': np.asarray(groups), 'signed': signed, 'abs': np.abs(signed)})
    df['within'] = 100 * (df['abs'] <= 10)
    return (df.groupby('group', observed=True, sort=False)
              .agg(homes=('abs', 'size'), MAPE=('abs', 'mean'), MdAPE=('abs', 'median'),
                   **{'within 10%': ('within', 'mean')}, **{'median signed error': ('signed', 'median')}))


def bootstrap_intervals(y_true, y_pred, n_resamples=1000, seed=prep.RANDOM_STATE):
    """95% bootstrap ranges for the main measures of one model.

    The evaluation homes are resampled with replacement and each measure is
    recomputed; returns a table with the observed value and the 2.5th and
    97.5th percentiles for R2, MAPE, MdAPE, MAE, RMSE and the share within 10%.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_resamples, len(y_true)))
    yt, yp = y_true[idx], y_pred[idx]
    err = yp - yt
    ape = np.abs(err) / yt
    ss_res = (err ** 2).sum(axis=1)
    ss_tot = ((yt - yt.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    draws = {
        'R2': 1 - ss_res / ss_tot,
        'MAPE': 100 * ape.mean(axis=1),
        'MdAPE': 100 * np.median(ape, axis=1),
        'MAE': np.abs(err).mean(axis=1),
        'RMSE': np.sqrt((err ** 2).mean(axis=1)),
        'within 10%': 100 * (ape <= 0.10).mean(axis=1),
    }
    observed = full_metrics(y_true, y_pred)
    return pd.DataFrame({m: {'value': observed[m], '95% low': np.percentile(d, 2.5), '95% high': np.percentile(d, 97.5)}
                         for m, d in draws.items()}).T
