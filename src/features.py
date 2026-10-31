"""Week 6 feature engineering.

Every feature here is computed row by row from a home's own record (or from a
fixed public map), so nothing is learned from other homes and the features can
be added to all months before splitting. Anything learned from data, such as
the encoding of school districts, happens later inside the fitted preprocessor.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src import preprocessing as prep

SCHOOL_DISTRICTS_SHP = (Path(__file__).resolve().parents[1] / 'data' / 'external' / 'school_districts_2425'
                        / 'DistrictAreas2425.shp')

# Feature sets compared in Week 6. Age at sale replaces year built: the two carry
# nearly the same information, and the best-practices document recommends age.
ORIGINAL = list(prep.FEATURES)
WITH_ENGINEERED = [c for c in ORIGINAL if c != 'YearBuilt'] + prep.ENGINEERED_FEATURES
WITH_TIME = WITH_ENGINEERED + prep.TIME_FEATURES
WITH_SCHOOLS = WITH_TIME + prep.DISTRICT_FEATURES
WITH_NEARBY = WITH_SCHOOLS + [prep.NEARBY_FEATURE]
FEATURE_SETS = {
    'Original (Week 3)': ORIGINAL,
    '+ age, bed/bath ratio': WITH_ENGINEERED,
    '+ month of sale': WITH_TIME,
    '+ school districts': WITH_SCHOOLS,
    '+ nearby sales price per sq ft': WITH_NEARBY,
}


def add_engineered_features(df):
    """Age at sale, bedroom-to-bathroom ratio and month-of-sale features."""
    df = df.copy()
    year = df['CloseDate'].dt.year
    month = df['CloseDate'].dt.month
    # A home sold in the year before its recorded completion year gets age 0, not -1
    df['AgeAtSale'] = (year - df['YearBuilt']).clip(lower=0)
    df['BedBathRatio'] = df['BedroomsTotal'] / df['BathroomsTotalInteger']
    # Month as a point on a circle, so December (12) and January (1) are neighbors
    df['SaleMonthSin'] = np.sin(2 * np.pi * month / 12)
    df['SaleMonthCos'] = np.cos(2 * np.pi * month / 12)
    # Months since January 2024, so a model can tell older sales from recent ones
    df['SaleMonthIndex'] = (year - 2024) * 12 + (month - 1)
    return df


def load_school_districts(path=SCHOOL_DISTRICTS_SHP):
    """District boundaries with only the identifying columns. The file's demographic
    columns (race and ethnicity shares, disadvantaged-student shares) are deliberately
    not used: neighborhood demographics are not valid inputs for a home-price model."""
    districts = gpd.read_file(path)[['CDSCode', 'DistrictNa', 'DistrictTy', 'geometry']]
    return districts.rename(columns={'DistrictNa': 'DistrictName', 'DistrictTy': 'DistrictType'})


def add_school_districts(df, districts=None):
    """Attach each home's school districts by locating its coordinates on the map.

    The map combines unified (all grades), elementary and high-school districts,
    which overlap. Each home gets two columns, both holding a district code:
    SchoolDistrictElem, its unified district if it has one, else its elementary
    district; SchoolDistrictHigh, its unified district if it has one, else its
    high-school district. Homes without coordinates, or outside every district,
    get missing values.
    """
    districts = load_school_districts() if districts is None else districts
    df = df.copy()
    has_xy = df['Latitude'].notna() & df['Longitude'].notna()
    points = gpd.GeoDataFrame(
        {'row': df.index[has_xy]},
        geometry=gpd.points_from_xy(df.loc[has_xy, 'Longitude'], df.loc[has_xy, 'Latitude'], crs='EPSG:4326'),
    ).to_crs(districts.crs)
    hits = gpd.sjoin(points, districts, how='inner', predicate='within')
    # One code per home and district type (the first, if boundaries overlap by mistake)
    by_type = hits.sort_values('CDSCode').groupby(['row', 'DistrictType'])['CDSCode'].first().unstack()
    for t in ['Unified', 'Elementary', 'High']:
        if t not in by_type:
            by_type[t] = np.nan
    elem = by_type['Unified'].fillna(by_type['Elementary'])
    high = by_type['Unified'].fillna(by_type['High'])
    df['SchoolDistrictElem'] = elem.reindex(df.index).astype(object).where(lambda s: s.notna(), np.nan)
    df['SchoolDistrictHigh'] = high.reindex(df.index).astype(object).where(lambda s: s.notna(), np.nan)
    return df


def add_all_features(clean_df):
    return add_school_districts(add_engineered_features(clean_df))
