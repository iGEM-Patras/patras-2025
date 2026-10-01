"""Download the two GEO inputs the preprocessing needs.

Fetches, into ../data/:
  GSE134358_series_matrix.txt   the expression series matrix (~200 MB decompressed)
  GPL21572.txt                  the platform annotation, probe ID -> miRNA name

Both are then read directly by preprocessor.py; no conversion is needed.
Files that are already present are left alone, so the script is safe to re-run.

Run from anywhere:  python biomarker_analysis/download_geo_data.py
"""

import gzip
import shutil
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

SERIES_GZ = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE134nnn/GSE134358/"
             "matrix/GSE134358_series_matrix.txt.gz")
PLATFORM = ("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
            "?acc=GPL21572&targ=self&form=text&view=data")


def fetch(url, dest, decompress=False):
    if dest.exists():
        print(f"  {dest.name} already present ({dest.stat().st_size/1e6:.0f} MB), skipping")
        return
    print(f"  downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out)
    if decompress:
        with gzip.open(tmp, "rb") as gz, open(dest, "wb") as out:
            shutil.copyfileobj(gz, out)
        tmp.unlink()
    else:
        tmp.replace(dest)
    print(f"  wrote {dest} ({dest.stat().st_size/1e6:.0f} MB)")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {DATA}")
    fetch(SERIES_GZ, DATA / "GSE134358_series_matrix.txt", decompress=True)
    fetch(PLATFORM, DATA / "GPL21572.txt")
    print("\nDone. Next: python biomarker_analysis/preprocessor.py")


if __name__ == "__main__":
    main()
