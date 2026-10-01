"""
R3.2 — regenerates manuscript Figure 2 (SHAP summary plots) with the alignment bug fixed.

The original figure plotted per-fold attribution values against the full dataset in its
ORIGINAL sample order, so row i of the SHAP matrix did not correspond to row i of the
feature matrix. The colour encoding was therefore scrambled, which is why the manuscript
described Random Forest and XGBoost as showing "a clear overlap between samples with low
and high biomarker values". That overlap was the plotting error, not a property of the
models.

Fix: concatenate the held-out feature rows in the same fold order as the SHAP values.

Colour: single-hue sequential ramp, light -> dark, for feature value (a magnitude
encoding). This departs from SHAP's default blue-red default, which is a two-hue ramp
without a neutral midpoint; the colourbar is labelled explicitly.
"""
import numpy as np, pandas as pd, warnings, ast, matplotlib
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
B    = str(REPO) + '/'
OUT  = str(REPO / 'results' / 'biomarker_analysis') + '/'
Path(OUT).mkdir(parents=True, exist_ok=True)

RS = 42
TB = ['hsa-miR-150-5p', 'hsa-miR-486-5p']

d = pd.read_excel(B + 'data/pruned_preprocessed_data.xlsx', header=None)
y = d.iloc[0, 1:].values.astype(int)
names = list(d.iloc[1:, 0].values)
X = d.iloc[1:, 1:].T.values.astype(float)[:, [names.index(t) for t in TB]]
Xs = StandardScaler().fit_transform(X)

# the F1-optimised configurations reported in manuscript Table 3
cfg = pd.read_excel(B + 'results/biomarker_analysis/f1_ranking/biomarker_model_results_f1_ranking.xlsx')
P = {r['Model']: ast.literal_eval(r['Best_Parameters']) for _, r in cfg.iterrows()}

MODELS = {
    'MLP':           (MLPClassifier(random_state=RS, max_iter=2000, **P['MLP']), True),
    'SVM':           (SVC(probability=True, random_state=RS, **P['SVM']), True),
    'Random Forest': (RandomForestClassifier(random_state=RS, **P['Random Forest']), False),
    'XGBoost':       (XGBClassifier(eval_metric='logloss', random_state=RS, **P['XGBoost']), False),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RS)

def aligned_shap(model, data, tree):
    """Return (shap_values, features) with rows in matching fold order."""
    S, F = [], []
    for tr, te in cv.split(data, y):
        m = model.__class__(**model.get_params())
        m.fit(data[tr], y[tr])
        if tree:
            sv = shap.TreeExplainer(m).shap_values(data[te])
            if isinstance(sv, list):        sv = sv[1]
            elif sv.ndim == 3:              sv = sv[:, :, 1]
        else:
            bg = shap.kmeans(data[tr], 10)
            sv = shap.KernelExplainer(lambda z: m.predict_proba(z)[:, 1], bg).shap_values(data[te])
        S.append(np.asarray(sv)); F.append(data[te])      # <- the fix: features follow the folds
    return np.concatenate(S), np.concatenate(F)

# single-hue sequential ramp, light -> dark
CMAP = LinearSegmentedColormap.from_list('teal', ['#D7EAE8', '#3C9490', '#0B4F4C'])

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.edgecolor': '#B8B8B8', 'axes.linewidth': 0.6,
                     'xtick.color': '#666', 'ytick.color': '#666'})
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), dpi=300)
fig.subplots_adjust(hspace=0.55, wspace=0.34, left=0.135, right=0.86, top=0.94, bottom=0.11)
rng = np.random.default_rng(0)
report = []

for idx, (ax, (letter, name)) in enumerate(zip(axes.ravel(), zip('ABCD', MODELS))):
    model, scaled = MODELS[name]
    data = Xs if scaled else X
    S, F = aligned_shap(model, data, name in ('Random Forest', 'XGBoost'))
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#EAEAEA', linewidth=0.6)
    ax.axvline(0, color='#9A9A9A', linewidth=0.9)
    for row, feat in enumerate(TB):
        v = F[:, row]
        norm = (v - v.min()) / (v.max() - v.min() + 1e-12)
        jitter = rng.uniform(-0.16, 0.16, len(v))
        ax.scatter(S[:, row], np.full(len(v), 1 - row) + jitter, c=norm, cmap=CMAP,
                   s=9, linewidths=0, alpha=0.85, rasterized=True)
        report.append((name, feat, np.corrcoef(F[:, row], S[:, row])[0, 1]))
    ax.set_yticks([1, 0])
    ax.set_yticklabels(TB if idx % 2 == 0 else ['', ''], fontsize=8.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(f'({letter}) {name}', fontsize=9.5, loc='left', pad=6)
    for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0)
    if idx >= 2:
        ax.set_xlabel('SHAP value  (positive drives the prediction toward sepsis)', fontsize=8)

cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), ax=axes, fraction=0.020, pad=0.025)
cbar.set_label('miRNA expression  (low \u2192 high)', fontsize=8.5)
cbar.set_ticks([0, 1]); cbar.set_ticklabels(['low', 'high']); cbar.ax.tick_params(length=0)
cbar.outline.set_visible(False)

for ext in ('png', 'pdf'):
    fig.savefig(OUT + f'Fig2_SHAP_corrected.{ext}', dpi=300, facecolor='white')
print('wrote Fig2_SHAP_corrected.png / .pdf\n')
print(f'{"model":16s} {"marker":18s} corr(expression, own SHAP)')
print('-' * 62)
for m, f, c in report:
    print(f'{m:16s} {f:18s} {c:+.3f}')
print('\nall negative => low expression drives the prediction toward sepsis, for both markers')
