# Aechmi — biomarker analysis and kinetic simulations

Code, inputs and results for the computational analyses of Aechmi.

The deposit covers two independent analyses:

1. **Biomarker selection** — identification of the hsa-miR-150-5p / hsa-miR-486-5p
   pair from GEO series GSE134358, with leak-free cross-validation.
2. **Kinetic simulations** — the catalytic hairpin assembly (CHA) rate constants,
   the CHA strand-displacement ODE model, and the coupled Cas13a-CHA model behind
   Figure 7.

Everything needed to re-run both is here except the raw GEO data, which is too
large to deposit and is downloaded as described below.

## Layout

```
.
├── data/                   inputs (small; the large GEO file is downloaded)
├── biomarker_analysis/     Python, biomarker selection pipeline
├── kinetic_models/         Python + R, rate constants and ODE models
└── results/                outputs reported in the manuscript
```

## Software

Analyses were run on Windows 11 with:

| | version |
|---|---|
| Python | 3.10.9 |
| numpy | 2.1.3 |
| scipy | 1.15.3 |
| pandas | 2.3.0 |
| matplotlib | 3.10.3 |
| scikit-learn | 1.7.0 |
| xgboost | 3.0.3 |
| shap | 0.48.0 |
| openpyxl | 3.1.5 |

```bash
pip install -r requirements.txt
```

Two additional dependencies are not installable from PyPI:

- **NUPACK 4** (`kinetic_models/forward_rate_calculation.py`,
  `reverse_rates_calculation.py`) is distributed under an academic licence from
  <https://nupack.org> and must be requested and installed separately.
- **R** with `deSolve` and `ggplot2` (`kinetic_models/figure7_cas13a_cha.R`):
  `install.packages(c("deSolve", "ggplot2"))`.

## Obtaining the GEO data

The biomarker analysis starts from GEO series **GSE134358** (Affymetrix miRNA 4.0,
platform GPL21572; 299 samples, 158 sepsis / 141 non-sepsis).

```bash
python biomarker_analysis/download_geo_data.py
```

That fetches both inputs into `data/`, decompresses the series matrix, and skips
anything already downloaded:

| file | size | source |
|---|---|---|
| `GSE134358_series_matrix.txt` | 192 MB | <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE134nnn/GSE134358/matrix/GSE134358_series_matrix.txt.gz> |
| `GPL21572.txt` | 97 MB | <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL21572&targ=self&form=text&view=data> |

Both are read as downloaded — no conversion to Excel is needed. `preprocessor.py`
also accepts previously converted `.xlsx` versions (`GSE134358_series_matrix.xlsx`,
`GPL21572-124634.xlsx`) if you already have them. Neither file is deposited here;
both are excluded in `.gitignore`.

Note that the script slices the series matrix by absolute row position — row 37
is the disease-state line, row 63 the first probe — so it is tied to the file
layout as distributed at the time of the analysis. On that file the parsed table
is 36,249 probes × 567 samples, which the disease-state filter reduces to the 299
samples used here (158 sepsis, 141 non-sepsis).

The preprocessing chain was verified end to end for this deposit: downloading both
files and running `preprocessor.py` followed by `feature_pruner.py` regenerates
`data/pruned_preprocessed_data.xlsx` identically to the deposited copy — same 263
miRNA names, same class labels, and values agreeing to 1.8e-15, which is Excel
round-tripping noise.

Readers who only want to reproduce the modelling and not the preprocessing can
skip the download entirely: `data/pruned_preprocessed_data.xlsx` is deposited and
every downstream script starts from it.

## Running the biomarker analysis

Run from the repository root, in this order. The first three steps need the GEO
download; the rest start from the deposited `pruned_preprocessed_data.xlsx`.

```bash
python biomarker_analysis/download_geo_data.py       # fetch the two GEO inputs
python biomarker_analysis/preprocessor.py            # GEO matrix -> preprocessed_data.xlsx
python biomarker_analysis/feature_pruner.py          # keep the 263 curated miRNAs
python biomarker_analysis/feature_ranker.py          # XGBoost + SHAP screen of all pairs
python biomarker_analysis/chosen_biomarkers_test.py  # four models on the selected pair
python biomarker_analysis/decision_boundary_visualizer.py
python biomarker_analysis/shap_figure.py             # manuscript Figure 2
python biomarker_analysis/leakfree_cv_analysis.py    # Tables 1 and 2, nested CV
python biomarker_analysis/rank_robustness.py         # rank stability across classifiers
```

`chosen_biomarkers_test.py` standardises the full matrix before its own
cross-validation loop, so its metrics are mildly optimistic, and its SHAP summary
plots mis-align the attribution rows against the feature rows. Both are stated in
the file itself. It is kept because the Figure 1 panels are drawn from its output
and is deposited unchanged so those stay reproducible; neither its metrics nor its
SHAP plots are the ones reported in the manuscript.

`shap_figure.py` produces manuscript Figure 2. It fits each model inside the
cross-validation folds and computes SHAP values on the held-out rows only, then
concatenates the feature rows in the same fold order — the alignment fix that the
earlier script lacked. Re-running it reproduces the deposited
`Fig2_SHAP_corrected.png` byte for byte, identical to the figure embedded in the
revised manuscript.

`leakfree_cv_analysis.py` is the leak-free estimate reported in the manuscript:
logistic regression inside a `Pipeline` with scaling, 10-fold stratified CV, and
nested CV where stated, so no information crosses the fold boundary. It writes
`Table1_regenerated.xlsx`, `Table2_regenerated.xlsx` (all 34,453 pairs) and
`r12_results.json`.

`rank_robustness.py` reads `Table2_regenerated.xlsx`, so run it after
`leakfree_cv_analysis.py`.

## Running the kinetic simulations

```bash
python kinetic_models/forward_rate_calculation.py    # needs NUPACK
python kinetic_models/reverse_rates_calculation.py   # needs NUPACK
python kinetic_models/strand_displacement_ode_model.py
Rscript kinetic_models/figure7_cas13a_cha.R
```

- `strand_displacement_ode_model.py` — the two-step CHA cascade alone
  (initiator + H1 ⇌ I:H1, then I:H1 + H2 ⇌ H1:H2 + initiator), at 1 µM hairpins
  and 100 nM initiator, over 50 s.
- `figure7_cas13a_cha.R` — the model behind Figure 7: the same CHA cascade driven
  by a Michaelis-Menten Cas13a term that converts H0 into initiator, solved with
  `deSolve` for healthy (1 pM) and septic (5 pM) microRNA levels over one hour.
  Tolerances are set tightly (`rtol = 1e-8`, `atol = 1e-18`) because the state
  variables span 1e-12 to 1e-6 M and the solver defaults are far too loose at that
  scale. Uncomment the final `ggsave` line to write the figure to disk.

## Data provenance

| file | origin |
|---|---|
| `data/pruned_preprocessed_data.xlsx` | GSE134358, preprocessed and restricted to 263 literature-curated miRNAs |
| `data/bibliography_biomarkers.xlsx` | the 263 curated miRNAs and their source publications |
| `data/sample_mapping_reference.csv` | GSE134358 sample → sepsis / non-sepsis label |
| `data/GSE134358_age_sex_by_sample.tsv` | age and sex per sample, from the GEO sample characteristics |

## Licence

Software (`biomarker_analysis/`, `kinetic_models/`) is released under the **MIT
licence**; data and results (`data/`, `results/`) under **CC BY 4.0**, matching the
licence of the associated article. Files derived from GEO series GSE134358 remain
subject to the terms of that original deposit and should be cited alongside this
repository. See `LICENSE` for the full text.

## Citation

`CITATION.cff` carries the machine-readable citation metadata; GitHub renders a
"Cite this repository" button from it. Once the associated article is published,
add it there as a `preferred-citation` so that citers are pointed at the article
as well as at this deposit.
