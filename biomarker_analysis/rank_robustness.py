"""
R1.2 / R3.2 supporting analysis: is the rank of the selected pair a stable property
of the data, or an artefact of the classifier used to produce the ranking?

Re-ranks the leading 150 pairs (from the logistic-regression ranking) under six
classifier families and records where the selected pair lands in each.

Result: rank spans 15-45 while the pair's AUC moves by only 0.027, and four
different pairs take first place. Every one of them contains hsa-miR-150-5p.
"""
import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

from pathlib import Path
B = str(Path(__file__).resolve().parent.parent) + '/'
d = pd.read_excel(B+'data/pruned_preprocessed_data.xlsx', header=None)
y = d.iloc[0,1:].values.astype(int); names = list(d.iloc[1:,0].values)
X = d.iloc[1:,1:].T.values.astype(float)
T2 = pd.read_excel(B+'results/biomarker_analysis/Table2_regenerated.xlsx')
SEL = {'hsa-miR-150-5p', 'hsa-miR-486-5p'}

pool = []
for f in T2.head(150)['Features']:
    a, b = [s.strip() for s in f.split(', ')]
    pool.append((names.index(a), names.index(b)))
si = (names.index('hsa-miR-150-5p'), names.index('hsa-miR-486-5p'))
if not any(set([names[i], names[j]]) == SEL for i, j in pool):
    pool.append(si)
sel_pos = [k for k, (i, j) in enumerate(pool) if set([names[i], names[j]]) == SEL][0]

MODELS = {
 'Logistic regression (used for the tables)':
     Pipeline([('s',StandardScaler()),('c',LogisticRegression(max_iter=5000))]),
 'Logistic regression, C=0.1':
     Pipeline([('s',StandardScaler()),('c',LogisticRegression(C=0.1,max_iter=5000))]),
 'SVM, RBF kernel':
     Pipeline([('s',StandardScaler()),('c',SVC(probability=True,random_state=42))]),
 'k-nearest neighbours, k=15':
     Pipeline([('s',StandardScaler()),('c',KNeighborsClassifier(15))]),
 'Random forest':
     Pipeline([('c',RandomForestClassifier(n_estimators=200,min_samples_leaf=3,
                                           random_state=42,n_jobs=-1))]),
 'Gradient boosting':
     Pipeline([('c',GradientBoostingClassifier(n_estimators=100,max_depth=2,random_state=42))]),
}
cv = StratifiedKFold(10, shuffle=True, random_state=42)
print(f'candidate pool: {len(pool)} pairs\n')
print(f"{'classifier':42s} {'sel AUC':>8s} {'rank':>10s} {'best AUC':>9s}  best pair")
print('-'*112)
ranks, aucs, winners = [], [], []
for nm, est in MODELS.items():
    sc = np.array([roc_auc_score(y, cross_val_predict(est, X[:,[i,j]], y, cv=cv,
                   method='predict_proba')[:,1]) for i, j in pool])
    order = np.argsort(-sc)
    r = int(np.where(order == sel_pos)[0][0]) + 1
    bi, bj = pool[order[0]]
    ranks.append(r); aucs.append(sc[sel_pos]); winners.append((names[bi], names[bj]))
    print(f'{nm:42s} {sc[sel_pos]:8.3f} {r:>7d}/{len(pool)} {sc[order[0]]:9.3f}  '
          f'{names[bi]} + {names[bj]}')
print('-'*112)
print(f'rank of the selected pair spans {min(ranks)} to {max(ranks)} '
      f'({max(ranks)/min(ranks):.1f}x) across classifier families')
print(f'its AUC spans only {min(aucs):.3f} to {max(aucs):.3f} (range {max(aucs)-min(aucs):.3f})')
uniq = {tuple(sorted(w)) for w in winners}
print(f'{len(uniq)} different pairs take first place across {len(winners)} families; '
      f'all contain hsa-miR-150-5p: '
      f'{all("hsa-miR-150-5p" in w for w in uniq)}')
