"""
R1.2 — regenerates Tables 1 and 2 with a properly specified estimator, computes the
nested-CV performance of the selected pair, and produces the panel-size / comparator
ceiling analysis.

Why the estimator changed: the original tables used XGBoost at default hyperparameters
on one- and two-feature problems, which overfits and depresses every value. Logistic
regression is the appropriate model at this dimensionality and is what a reviewer would
expect. All estimates are leak-free (10-fold stratified CV; nested CV where stated).
"""
import pandas as pd, numpy as np, warnings, json, sys, time
warnings.filterwarnings('ignore')
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, f1_score

RS = 42
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
BASE = str(REPO) + '/'
OUT  = str(REPO / 'results' / 'biomarker_analysis') + '/'
Path(OUT).mkdir(parents=True, exist_ok=True)

d = pd.read_excel(BASE+'data/pruned_preprocessed_data.xlsx', header=None)
y = d.iloc[0,1:].values.astype(int)
names = list(d.iloc[1:,0].values)
X = d.iloc[1:,1:].T.values.astype(float)
grp = (pd.read_csv(BASE+'data/sample_mapping_reference.csv')
         ['Original_Label'].str.replace('disease state: ','',regex=False).values)
assert ((grp=='sepsis').astype(int)==y).all()
i150, i486 = names.index('hsa-miR-150-5p'), names.index('hsa-miR-486-5p')
cv10 = StratifiedKFold(10, shuffle=True, random_state=RS)
rng  = np.random.default_rng(0)
t0 = time.time()

def lr():
    return Pipeline([('s',StandardScaler()),('c',LogisticRegression(max_iter=5000))])

def cv_scores(cols, yy=None, XX=None):
    yy = y if yy is None else yy
    XX = X if XX is None else XX
    p = cross_val_predict(lr(), XX[:,cols], yy, cv=cv10, method='predict_proba')[:,1]
    return roc_auc_score(yy,p), f1_score(yy,(p>=0.5).astype(int)), p

def boot_ci(yy, pp, n=3000):
    bs=[]
    for _ in range(n):
        b=rng.integers(0,len(yy),len(yy))
        if len(np.unique(yy[b]))>1: bs.append(roc_auc_score(yy[b],pp[b]))
    return np.percentile(bs,2.5), np.percentile(bs,97.5)

# ------------------------------------------------------------------ TABLE 1
print('[1/5] Table 1 — all 263 single markers ...'); sys.stdout.flush()
rows=[]
for j,n in enumerate(names):
    a,f,_ = cv_scores([j])
    rows.append(dict(Features=n, ROC_AUC=a, F1_Score=f))
t1 = pd.DataFrame(rows).sort_values('ROC_AUC', ascending=False).reset_index(drop=True)
t1.insert(0,'Rank',t1.index+1)
t1.to_excel(OUT+'Table1_regenerated.xlsx', index=False)
print(t1.head(15).to_string(index=False, float_format=lambda v:f'{v:.3f}'))
print(f"\n  single-marker AUC range across all 263: {t1.ROC_AUC.min():.3f} - {t1.ROC_AUC.max():.3f}")
print(f"  top-15 AUC range: {t1.ROC_AUC.iloc[14]:.3f} - {t1.ROC_AUC.iloc[0]:.3f}")
print(f"  hsa-miR-150-5p rank {int(t1[t1.Features=='hsa-miR-150-5p'].Rank.iloc[0])}, "
      f"AUC {t1[t1.Features=='hsa-miR-150-5p'].ROC_AUC.iloc[0]:.3f}")
print(f"  hsa-miR-486-5p rank {int(t1[t1.Features=='hsa-miR-486-5p'].Rank.iloc[0])}, "
      f"AUC {t1[t1.Features=='hsa-miR-486-5p'].ROC_AUC.iloc[0]:.3f}")

# ------------------------------------------------------------------ TABLE 2
print(f'\n[2/5] Table 2 — all {263*262//2:,} pairs ...'); sys.stdout.flush()
pairs = list(combinations(range(len(names)),2))
rows=[]
for k,(a,b) in enumerate(pairs):
    au,f,_ = cv_scores([a,b])
    rows.append((f'{names[a]}, {names[b]}', au, f))
    if (k+1)%5000==0:
        print(f'    {k+1:,}/{len(pairs):,}  ({time.time()-t0:.0f}s)'); sys.stdout.flush()
t2 = pd.DataFrame(rows, columns=['Features','ROC_AUC','F1_Score']).sort_values(
        'ROC_AUC',ascending=False).reset_index(drop=True)
t2.insert(0,'Rank',t2.index+1)
t2.to_excel(OUT+'Table2_regenerated.xlsx', index=False)
print(t2.head(15).to_string(index=False, float_format=lambda v:f'{v:.3f}'))
sel = t2[t2.Features.str.contains('150-5p') & t2.Features.str.contains('486-5p')]
sel_rank = int(sel.Rank.iloc[0])
print(f"\n  SELECTED PAIR rank {sel_rank} of {len(t2):,}   AUC {sel.ROC_AUC.iloc[0]:.3f}"
      f"   (percentile {100*(1-sel_rank/len(t2)):.2f})")

# ------------------------------------------------------------------ NESTED CV
print('\n[3/5] Nested CV for the selected pair ...'); sys.stdout.flush()
GRIDS = {
 'Logistic regression': (lr(), {'c__C':[0.01,0.1,1,10,100]}),
 'SVM (RBF)': (Pipeline([('s',StandardScaler()),('c',SVC(probability=True,random_state=RS))]),
               {'c__C':[0.1,1,10,100],'c__gamma':['scale',0.01,0.1,1]}),
 'Random forest': (Pipeline([('c',RandomForestClassifier(random_state=RS,n_jobs=-1))]),
                   {'c__n_estimators':[300],'c__max_depth':[3,5,None],'c__min_samples_leaf':[1,3,5]}),
}
nested={}
for nm,(est,g) in GRIDS.items():
    oof=np.zeros(len(y))
    for tr,te in cv10.split(X[:,[i150,i486]],y):
        gs=GridSearchCV(est,g,scoring='roc_auc',
                        cv=StratifiedKFold(5,shuffle=True,random_state=RS),n_jobs=-1)
        gs.fit(X[tr][:,[i150,i486]],y[tr])
        oof[te]=gs.best_estimator_.predict_proba(X[te][:,[i150,i486]])[:,1]
    a=roc_auc_score(y,oof); lo,hi=boot_ci(y,oof)
    nested[nm]=(a,lo,hi,oof)
    print(f'   {nm:22s} AUC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]')
best=max(nested,key=lambda k:nested[k][0])
print(f'   -> headline: {best}, AUC {nested[best][0]:.3f} '
      f'[{nested[best][1]:.3f}, {nested[best][2]:.3f}]')

# ------------------------------------------------------------------ CEILING + COMPARATOR
print('\n[4/5] Panel-size ceiling, split by comparator ...'); sys.stdout.flush()
M_all = np.ones(len(y),bool)
M_hv  = (grp=='sepsis')|(grp=='healthy')
M_ni  = (grp=='sepsis')|(grp=='noninfectious')
KS = [2,5,10,30,100,263]
ceiling={}
for lbl,M in [('all',M_all),('healthy',M_hv),('noninfectious',M_ni)]:
    yy,XX = y[M], X[M]
    row={}
    p = cross_val_predict(lr(),XX[:,[i150,i486]],yy,cv=cv10,method='predict_proba')[:,1]
    row['pair']=(roc_auc_score(yy,p),)+boot_ci(yy,p)
    for k in KS:
        pipe=Pipeline([('sel',SelectKBest(f_classif,k=k)),('s',StandardScaler()),
                       ('c',LogisticRegression(max_iter=5000))])
        pp=cross_val_predict(pipe,XX,yy,cv=cv10,method='predict_proba')[:,1]
        row[k]=(roc_auc_score(yy,pp),)+boot_ci(yy,pp)
    ceiling[lbl]=row
    print(f"   {lbl:14s} pair {row['pair'][0]:.3f} | " +
          " ".join(f"k{k}:{row[k][0]:.3f}" for k in KS))

# ------------------------------------------------------------------ INCREMENT
print('\n[5/5] Incremental value of the second marker ...'); sys.stdout.flush()
_,_,p1 = cv_scores([i150]); _,_,p2 = cv_scores([i150,i486])
a1,a2 = roc_auc_score(y,p1), roc_auc_score(y,p2)
dd=[]
for _ in range(5000):
    b=rng.integers(0,len(y),len(y))
    if len(np.unique(y[b]))>1: dd.append(roc_auc_score(y[b],p2[b])-roc_auc_score(y[b],p1[b]))
dd=np.array(dd)
print(f'   miR-150 alone {a1:.3f} -> pair {a2:.3f}   gain {a2-a1:+.3f} '
      f'[{np.percentile(dd,2.5):+.3f}, {np.percentile(dd,97.5):+.3f}] p={2*min((dd<=0).mean(),(dd>=0).mean()):.3f}')

json.dump({'nested':{k:[v[0],v[1],v[2]] for k,v in nested.items()},
           'ceiling':{k:{str(kk):list(vv) for kk,vv in v.items()} for k,v in ceiling.items()},
           'sel_rank':sel_rank,'n_pairs':len(t2),
           'sel_auc':float(sel.ROC_AUC.iloc[0]),
           't1_top':t1.head(15).to_dict('records')},
          open(OUT+'r12_results.json','w'), indent=1)
np.save(OUT+'oof_best.npy', nested[best][3])
print(f'\ndone in {time.time()-t0:.0f}s -> Table1_regenerated.xlsx, Table2_regenerated.xlsx, r12_results.json')
