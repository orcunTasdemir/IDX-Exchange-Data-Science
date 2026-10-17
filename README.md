# IDX Exchange Data Science: California Property Close Price Prediction

Predicting `ClosePrice` (final sale price) for California single-family homes from CRMLS MLS data, as part of the IDX Exchange Data Science internship.

## Repository layout

| Path | Contents |
|---|---|
| `py/` | Week 1 Trestle API pull scripts (`crmls_sold.py`, `crmls_listed.py`) |
| `notebooks/01_exploration.ipynb` | Week 2: exploratory data analysis of the sold data |
| `reports/01_exploration_memo.md` | Week 2: findings memo ([read it here](reports/01_exploration_memo.md)) |
| `notebooks/02_preprocessing.ipynb` | Week 3: cleaning, encoding, chronological split, choice of training window |
| `reports/02_preprocessing_memo.md` | Week 3: preprocessing memo ([read it here](reports/02_preprocessing_memo.md)) |
| `notebooks/03_baseline_model.ipynb` | Week 4: linear regression baseline, test-month result, stability over time |
| `reports/03_baseline_memo.md` | Week 4: baseline memo ([read it here](reports/03_baseline_memo.md)) |
| `reports/03_meeting_memo.md` | Week 4: meeting memo, a shorter read-through version ([read it here](reports/03_meeting_memo.md)) |
| `scripts/loss_comparison_03.py` | Week 4 side check: squared vs absolute loss for the linear model |
| `src/preprocessing.py` | Cleaning, splitting and preprocessing code shared by all notebooks |
| `src/modeling.py` | Model wrapper (preprocessor + estimator on log price) and evaluation helpers |
| `scripts/run_notebooks.py` | Re-runs the notebooks reproducibly |
| `config/split_config.json` | Chosen training window, split months and outlier cutoffs (no data) |
| `results/` | Metric tables (scores only, committed), e.g. `03_baseline_metrics.csv` |
| `reports/figures/` | Charts produced by the notebooks |
| `requirements.txt` | Pinned Python environment |

## Data

Monthly `CRMLSSold<YYYYMM>.csv` and `CRMLSListing<YYYYMM>.csv` files from the IDX Exchange FTP (`/raw/California`). Field definitions follow the Trestle Property metadata. Current snapshot: sold closings from January 2024 through April 2026.

**Data is not committed.** Put the CSV files in a local `csv/` folder at the repo root. `.gitignore` blocks all data files.

`02_preprocessing.ipynb` writes the cleaned data to `data/processed/` (also not committed): `sfr_clean.csv` (all months after cleaning), `train.csv` (April 2025 to March 2026) and `test.csv` (April 2026).

## Re-running

```bash
pip install -r requirements.txt
python scripts/run_notebooks.py        # all notebooks, in order
python scripts/run_notebooks.py 02     # only 02_preprocessing.ipynb
```

**Reproducibility.** Every random step (target-encoding folds, gradient boosting) uses one seed, `RANDOM_STATE = 42` in `src/preprocessing.py`, and the notebooks also call `set_global_seed()`. The run script does not store execution timestamps, so rerunning unchanged code on unchanged data produces byte-identical notebooks, CSVs and charts. The one outside dependency is the street map in `01_exploration.ipynb`, which is downloaded from Esri on each run.

## Progress

- [x] Week 1: setup, data access
- [x] Week 2: exploration (`01_exploration.ipynb`, [findings memo](reports/01_exploration_memo.md))
- [x] Week 3: preprocessing (`02_preprocessing.ipynb`, cleaned CSVs, [memo](reports/02_preprocessing_memo.md))
- [x] Week 4: baseline model (`03_baseline_model.ipynb`, [memo](reports/03_baseline_memo.md)): test-month R² 0.820, MdAPE 11.38%
- [ ] Week 5: decision tree and random forest (`04_model_comparison.ipynb`)
