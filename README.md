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
| `src/preprocessing.py` | Cleaning, splitting and preprocessing code shared by all notebooks |
| `config/split_config.json` | Chosen training window, split months and outlier cutoffs (no data) |
| `reports/figures/` | Charts produced by the notebooks |
| `requirements.txt` | Pinned Python environment |

## Data

Monthly `CRMLSSold<YYYYMM>.csv` and `CRMLSListing<YYYYMM>.csv` files from the IDX Exchange FTP (`/raw/California`). Field definitions follow the Trestle Property metadata. Current snapshot: sold closings from January 2024 through April 2026.

**Data is not committed.** Put the CSV files in a local `csv/` folder at the repo root. `.gitignore` blocks all data files.

`02_preprocessing.ipynb` writes the cleaned data to `data/processed/` (also not committed): `sfr_clean.csv` (all months after cleaning), `train.csv` (April 2025 to March 2026) and `test.csv` (April 2026).

## Re-running

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
```

## Progress

- [x] Week 1: setup, data access
- [x] Week 2: exploration (`01_exploration.ipynb`, [findings memo](reports/01_exploration_memo.md))
- [x] Week 3: preprocessing (`02_preprocessing.ipynb`, cleaned CSVs, [memo](reports/02_preprocessing_memo.md))
- [ ] Week 4: baseline model (`03_baseline_model.ipynb`)
