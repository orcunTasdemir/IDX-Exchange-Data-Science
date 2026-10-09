"""Cleaning, splitting and feature preprocessing for the CRMLS close-price model.

The work is divided by whether a step needs to learn anything from data:

* ``clean`` applies fixed, row-by-row rules (deduplication, impossible values).
  It learns nothing, so it can run once on all months.
* ``make_split`` picks the training and evaluation months and applies price
  outlier cutoffs that are computed from the training months only.
* ``build_preprocessor`` returns an unfitted scikit-learn ColumnTransformer.
  Everything it learns (clipping bounds, medians, encodings, scaling) is
  learned when it is fit on the training rows, never on evaluation rows.
"""
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

# The one seed for the project. Every random step passes it explicitly
# (random_state=RANDOM_STATE); set_global_seed() also covers any code that
# draws from Python's or NumPy's global random generators.
RANDOM_STATE = 42


def set_global_seed(seed=RANDOM_STATE):
    """Seed Python's and NumPy's global random generators."""
    random.seed(seed)
    np.random.seed(seed)
    return seed


# California bounding box; coordinates outside it are treated as missing
CA_LAT = (32.4, 42.1)
CA_LON = (-124.6, -114.1)

# Model inputs, grouped by how they are preprocessed
LOG_FEATURES = ['LivingArea', 'LotSizeSquareFeet']          # right-skewed sizes
NUMERIC_FEATURES = ['BedroomsTotal', 'BathroomsTotalInteger', 'YearBuilt',
                    'GarageSpaces', 'Stories', 'AssociationFeeMonthly']
COORD_FEATURES = ['Latitude', 'Longitude']
BOOL_FEATURES = ['PoolPrivateYN', 'ViewYN', 'AttachedGarageYN', 'FireplaceYN', 'NewConstructionYN']
COUNTY_FEATURE = 'CountyOrParish'                             # about 60 values: one-hot
ZIP_FEATURE = 'ZIP5'                                          # about 3,000 values: target-encoded
FEATURES = LOG_FEATURES + NUMERIC_FEATURES + COORD_FEATURES + BOOL_FEATURES + [COUNTY_FEATURE, ZIP_FEATURE]

TARGET = 'ClosePrice'
ID_COLUMNS = ['ListingKey', 'CloseDate', 'CloseMonth', 'SourceMonth']

# Columns excluded because they are set by the sale process (see the AVM best-practices leakage table)
LEAKAGE_COLUMNS = ['ListPrice', 'OriginalListPrice', 'DaysOnMarket',
                   'BuyerAgencyCompensation', 'BuyerAgencyCompensationType']

FEE_PER_MONTH = {'Monthly': 1, 'Quarterly': 3, 'SemiAnnually': 6, 'Annually': 12}
LEVELS_TO_STORIES = {'One': 1, 'Two': 2, 'ThreeOrMore': 3}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_sold(data_dir):
    """Stack all CRMLSSold*.csv files; columns missing from a file become NaN."""
    files = sorted(Path(data_dir).glob('CRMLSSold*.csv'))
    if not files:
        raise FileNotFoundError(f'No CRMLSSold*.csv files in {data_dir}')
    frames = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d['SourceMonth'] = re.search(r'(\d{6})', f.name).group(1)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Row-level cleaning (fixed rules, nothing learned from data)
# ---------------------------------------------------------------------------
def _to_float_bool(s):
    return s.map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})


def clean(raw):
    """Apply fixed cleaning rules to the stacked raw data.

    Returns ``(clean_df, row_log, value_log)``. ``row_log`` records the rows
    removed by each step; ``value_log`` records values set to missing (they
    are filled later by the fitted preprocessor).
    """
    rows = []

    def log_rows(step, before, after):
        rows.append({'step': step, 'rows_before': before, 'rows_removed': before - after, 'rows_after': after})

    n = len(raw)
    df = raw[raw['PropertyType'] == 'Residential']
    log_rows('Keep PropertyType = Residential', n, len(df))
    n = len(df)
    df = df[df['PropertySubType'] == 'SingleFamilyResidence'].copy()
    log_rows('Keep PropertySubType = SingleFamilyResidence', n, len(df))

    for c in ['CloseDate', 'ListingContractDate', 'PurchaseContractDate']:
        df[c] = pd.to_datetime(df[c], errors='coerce')

    n = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != 'SourceMonth'])
    log_rows('Drop exact duplicate rows', n, len(df))

    # A repeated ListingKey is the same home recorded again in a later monthly file
    # (same address and living area); keep the most recent record.
    n = len(df)
    df = (df.sort_values(['ListingKey', 'SourceMonth', 'CloseDate'])
            .drop_duplicates('ListingKey', keep='last'))
    log_rows('Keep latest record per ListingKey', n, len(df))

    steps = [
        ('Drop ClosePrice missing or <= 0', ~(df['ClosePrice'] > 0)),
        ('Drop StateOrProvince other than CA', df['StateOrProvince'] != 'CA'),
        ('Drop CloseDate before ListingContractDate', df['CloseDate'] < df['ListingContractDate']),
        ('Drop CloseDate before PurchaseContractDate', df['CloseDate'] < df['PurchaseContractDate']),
        ('Drop LivingArea missing, < 300 or > 20,000 sqft', ~df['LivingArea'].between(300, 20_000)),
        ('Drop more than 15 bedrooms or bathrooms',
         (df['BedroomsTotal'] > 15) | (df['BathroomsTotalInteger'] > 15)),
    ]
    for step, condition in steps:
        # Conditions were evaluated before any of these removals; restricting each to the
        # rows still present makes the logged counts sequential
        mask = condition.reindex(df.index, fill_value=False)
        n = len(df)
        df = df[~mask]
        log_rows(step, n, len(df))

    # Value-level fixes: impossible values become missing and are imputed later
    values = []

    def set_missing(desc, col, mask):
        values.append({'fix': desc, 'column': col, 'values_set_missing': int(mask.sum()), 'values_filled': 0})
        df.loc[mask, col] = np.nan

    set_missing('0 bedrooms', 'BedroomsTotal', df['BedroomsTotal'] == 0)
    set_missing('0 bathrooms', 'BathroomsTotalInteger', df['BathroomsTotalInteger'] == 0)
    set_missing('YearBuilt before 1850 or after year of sale', 'YearBuilt',
                (df['YearBuilt'] < 1850) | (df['YearBuilt'] > df['CloseDate'].dt.year + 1))
    outside = df['Latitude'].notna() & ~(df['Latitude'].between(*CA_LAT) & df['Longitude'].between(*CA_LON))
    set_missing('Coordinates outside California', 'Latitude', outside)
    df.loc[outside, 'Longitude'] = np.nan
    set_missing('Lot size of 0 sqft', 'LotSizeSquareFeet', df['LotSizeSquareFeet'] == 0)

    # Value-level derivations: fill gaps from a second column holding the same fact
    lot_from_acres = df['LotSizeSquareFeet'].isna() & (df['LotSizeAcres'] > 0)
    df.loc[lot_from_acres, 'LotSizeSquareFeet'] = df.loc[lot_from_acres, 'LotSizeAcres'] * 43_560
    values.append({'fix': 'Lot size filled from LotSizeAcres x 43,560', 'column': 'LotSizeSquareFeet',
                   'values_set_missing': 0, 'values_filled': int(lot_from_acres.sum())})

    stories_from_levels = df['Stories'].isna() & df['Levels'].isin(LEVELS_TO_STORIES.keys())
    df.loc[stories_from_levels, 'Stories'] = df.loc[stories_from_levels, 'Levels'].map(LEVELS_TO_STORIES)
    values.append({'fix': 'Stories filled from Levels', 'column': 'Stories',
                   'values_set_missing': 0, 'values_filled': int(stories_from_levels.sum())})

    # Association fee expressed per month; a positive fee with no frequency is assumed monthly
    divisor = df['AssociationFeeFrequency'].map(FEE_PER_MONTH)
    divisor = divisor.where(divisor.notna() | ~(df['AssociationFee'] > 0), 1).fillna(1)
    df['AssociationFeeMonthly'] = df['AssociationFee'] / divisor

    for c in BOOL_FEATURES:
        df[c] = _to_float_bool(df[c])
    # 5-digit ZIP as a plain object column with np.nan for missing (what SimpleImputer expects)
    zip5 = df['PostalCode'].astype(str).str.extract(r'^(\d{5})')[0]
    df['ZIP5'] = zip5.astype(object).where(zip5.notna(), np.nan)
    df['CloseMonth'] = df['CloseDate'].dt.strftime('%Y-%m')

    clean_df = df[ID_COLUMNS + FEATURES + [TARGET]].reset_index(drop=True)
    row_log = pd.DataFrame(rows)
    row_log['pct_removed'] = (100 * row_log['rows_removed'] / row_log['rows_before']).round(3)
    return clean_df, row_log, pd.DataFrame(values)


# ---------------------------------------------------------------------------
# Chronological split with training-only outlier cutoffs
# ---------------------------------------------------------------------------
def fit_outlier_bounds(train, lower=0.005, upper=0.995):
    """Price and price-per-sqft cutoffs computed from training rows only."""
    ppsf = train[TARGET] / train['LivingArea']
    return {
        'price_low': float(train[TARGET].quantile(lower)),
        'price_high': float(train[TARGET].quantile(upper)),
        'ppsf_low': float(ppsf.quantile(lower)),
        'ppsf_high': float(ppsf.quantile(upper)),
    }


def apply_outlier_bounds(df, bounds):
    ppsf = df[TARGET] / df['LivingArea']
    keep = (df[TARGET].between(bounds['price_low'], bounds['price_high'])
            & ppsf.between(bounds['ppsf_low'], bounds['ppsf_high']))
    return df[keep]


def make_split(clean_df, eval_month, n_train_months):
    """Train on the ``n_train_months`` months right before ``eval_month``, evaluate on ``eval_month``.

    Outlier cutoffs are computed on the training months and applied unchanged
    to both sets. Returns ``(train, evaluation, bounds)``.
    """
    months = sorted(clean_df['CloseMonth'].unique())
    i = months.index(eval_month)
    if n_train_months > i:
        raise ValueError(f'Only {i} months available before {eval_month}')
    train_months = months[i - n_train_months:i]
    train = clean_df[clean_df['CloseMonth'].isin(train_months)]
    evaluation = clean_df[clean_df['CloseMonth'] == eval_month]
    bounds = fit_outlier_bounds(train)
    return apply_outlier_bounds(train, bounds), apply_outlier_bounds(evaluation, bounds), bounds


# ---------------------------------------------------------------------------
# Preprocessor (fit on training rows only)
# ---------------------------------------------------------------------------
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned at fit time (missing values pass through)."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class GroupMedianCoordImputer(BaseEstimator, TransformerMixin):
    """Fill missing coordinates with the training median of the home's ZIP code,
    then of its county, then of all training homes. Adds a 0/1 column marking
    which rows were filled.

    Input columns, in order: Latitude, Longitude, then the grouping columns.
    """

    def __init__(self, groups=('ZIP5', 'CountyOrParish')):
        self.groups = groups

    def _frame(self, X):
        return pd.DataFrame(np.asarray(X, dtype=object), columns=COORD_FEATURES + list(self.groups))

    def fit(self, X, y=None):
        X = self._frame(X)
        X[COORD_FEATURES] = X[COORD_FEATURES].astype(float)
        self.group_medians_ = [X.groupby(g)[COORD_FEATURES].median() for g in self.groups]
        self.overall_median_ = X[COORD_FEATURES].median()
        return self

    def transform(self, X):
        X = self._frame(X)
        out = X[COORD_FEATURES].astype(float)
        was_missing = out.isna().any(axis=1)
        for g, med in zip(self.groups, self.group_medians_):
            out = out.fillna(med.reindex(X[g]).set_axis(out.index))
        out = out.fillna(self.overall_median_)
        out['coords_imputed'] = was_missing.astype(float)
        return out.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array(COORD_FEATURES + ['coords_imputed'], dtype=object)


def build_preprocessor():
    """Unfitted ColumnTransformer turning the FEATURES columns into a numeric matrix."""
    log_numeric = Pipeline([
        ('clip', QuantileClipper()),
        ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ])
    numeric = Pipeline([
        ('clip', QuantileClipper()),
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ])
    coords = Pipeline([
        ('impute', GroupMedianCoordImputer()),
        ('scale', StandardScaler()),
    ])
    booleans = SimpleImputer(strategy='constant', fill_value=0, add_indicator=True)
    county = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=50, sparse_output=False)),
    ])
    zip_code = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('target', TargetEncoder(target_type='continuous',
                                  cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))),
        ('scale', StandardScaler()),
    ])
    return ColumnTransformer([
        ('log_numeric', log_numeric, LOG_FEATURES),
        ('numeric', numeric, NUMERIC_FEATURES),
        ('coords', coords, COORD_FEATURES + [ZIP_FEATURE, COUNTY_FEATURE]),
        ('booleans', booleans, BOOL_FEATURES),
        ('county', county, [COUNTY_FEATURE]),
        ('zip', zip_code, [ZIP_FEATURE]),
    ], verbose_feature_names_out=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    """R^2, MAPE, MdAPE (both in %) and MAE (dollars), all on the price scale."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        'R2': r2_score(y_true, y_pred),
        'MAPE': 100 * ape.mean(),
        'MdAPE': 100 * np.median(ape),
        'MAE': mean_absolute_error(y_true, y_pred),
    }
