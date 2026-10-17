"""Week 4 side check: squared vs absolute loss for the linear model, on dollars vs log price.

Trains on the settings split (12 months before March 2026) and scores the
March 2026 validation month; the test month is not used. The absolute-loss
fits take about 10 minutes each, which is why this runs as a script with
saved results instead of inside 03_baseline_model.ipynb.

Usage (from the repository root, after 02_preprocessing has been run):
    python scripts/loss_comparison_03.py
Writes results/03_loss_comparison.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.modeling as mdl  # noqa: E402
import src.preprocessing as prep  # noqa: E402

prep.set_global_seed()
config = json.loads((ROOT / 'config' / 'split_config.json').read_text())
clean_df = prep.load_clean(ROOT / 'data' / 'processed' / 'sfr_clean.csv')
validation_month = config['tuning']['validation_month']
train, val, _ = prep.make_split(clean_df, validation_month, config['n_train_months'])
y = val[prep.TARGET].to_numpy()
top_fifth = y >= np.quantile(y, 0.8)

variants = [
    ('squared loss', 'dollars', LinearRegression, False),
    ('absolute loss', 'dollars', lambda: QuantileRegressor(quantile=0.5, alpha=0, solver='highs'), False),
    ('squared loss', 'log price', LinearRegression, True),
    ('absolute loss', 'log price', lambda: QuantileRegressor(quantile=0.5, alpha=0, solver='highs'), True),
]
rows = []
for loss, target, make_estimator, log_target in variants:
    start = time.time()
    model = mdl.price_model(make_estimator(), log_target=log_target)
    model.fit(train[prep.FEATURES], train[prep.TARGET])
    pred = model.predict(val[prep.FEATURES])
    signed = 100 * (pred - y) / y
    rows.append({'loss': loss, 'target': target, 'eval_month': validation_month,
                 **prep.regression_metrics(y, pred),
                 'negative_predictions': int((pred <= 0).sum()),
                 'top_fifth_median_signed_error': float(np.median(signed[top_fifth])),
                 'fit_seconds': round(time.time() - start, 1)})
    print(rows[-1], flush=True)

out = ROOT / 'results' / '03_loss_comparison.csv'
pd.DataFrame(rows).to_csv(out, index=False, float_format='%.6f')
print('wrote', out.relative_to(ROOT))
