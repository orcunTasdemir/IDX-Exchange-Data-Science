# IDX Exchange Data Science: California Property Close Price Prediction

Predicting `ClosePrice` (final sale price) for California single-family homes from CRMLS MLS data, as part of the IDX Exchange Data Science internship.

## Repository layout

| Path | Contents |
|---|---|
| `py/` | Week 1 Trestle API pull scripts (`crmls_sold.py`, `crmls_listed.py`) |
| `notebooks/01_exploration.ipynb` | Week 2: exploratory data analysis of the sold data |
| `requirements.txt` | Pinned Python environment |

## Data

Monthly `CRMLSSold<YYYYMM>.csv` and `CRMLSListing<YYYYMM>.csv` files from the IDX Exchange FTP (`/raw/California`). Field definitions follow the Trestle Property metadata. Current snapshot: sold closings from January 2024 through April 2026.

**Data is not committed.** Put the CSV files in a local `csv/` folder at the repo root. `.gitignore` blocks all data files.

## Re-running

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploration.ipynb
```

## Progress

- [x] Week 1: setup, data access
- [x] Week 2: exploration (`01_exploration.ipynb`)
- [ ] Week 3: preprocessing (`02_preprocessing.ipynb`)
