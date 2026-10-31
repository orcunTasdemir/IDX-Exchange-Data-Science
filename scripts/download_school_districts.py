"""Download the California School District Areas 2024-25 boundaries (Week 6).

Source: California Department of Education, via the California open-data portal
(https://data.ca.gov/dataset/california-school-district-areas-2024-25),
listed as "No restrictions on public use". The file is saved to
data/external/ and is not committed.

Usage (from the repository root):
    python scripts/download_school_districts.py
"""
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = 'https://gis.data.ca.gov/api/download/v1/items/b0e3b936426a47ce9d9a2e77e2bb86cc/shapefile?layers=0'
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'external' / 'school_districts_2425'
ZIP_PATH = OUT_DIR.parent / 'DistrictAreas2425.zip'

OUT_DIR.mkdir(parents=True, exist_ok=True)
if not ZIP_PATH.exists():
    request = Request(URL, headers={'User-Agent': 'IDX-Exchange-Data-Science internship project'})
    with urlopen(request, timeout=300) as response:
        ZIP_PATH.write_bytes(response.read())
print(f'{ZIP_PATH.relative_to(ROOT)}: {ZIP_PATH.stat().st_size / 1e6:.1f} MB')

with zipfile.ZipFile(ZIP_PATH) as z:
    z.extractall(OUT_DIR)
    print('Extracted:', ', '.join(sorted(z.namelist())))
