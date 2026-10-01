import matplotlib
matplotlib.use('Agg')

import pandas as pd
import os
from pathlib import Path
from itertools import combinations
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import shap
import warnings
import time
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings(action='ignore', category=UserWarning, module='xgboost')
start_time = time.time()

base_path = Path(__file__).parent.parent / "data"
input_path = base_path / "pruned_preprocessed_data.xlsx"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Error: Input file not found at {input_path}")

data = pd.read_excel(input_path, header=None)
class_labels = data.iloc[0, 1:].values.astype(int)
feature_names = data.iloc[1:, 0].values
features = data.iloc[1:, 1:].T.values

model_name = "xgboost"
model_template = XGBClassifier(use_label_encoder=False, eval_metric='auc', n_jobs=-1)
combination_sizes = [1, 2] 

for size in combination_sizes:
    print(f"\nRunning {model_name} with combination size {size}")
    combinations_list = list(combinations(range(features.shape[1]), size))
    results = []

    result_file = base_path / f"{size}_{model_name}_results.xlsx"
    pics_folder = base_path / f"pics_{size}_{model_name}"
    os.makedirs(pics_folder, exist_ok=True)

    for i, feature_indices in enumerate(combinations_list, start=1):
        selected_features = features[:, feature_indices]
        feature_names_combination = [feature_names[idx] for idx in feature_indices]

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        f1_scores, roc_aucs = [], []

        for train_idx, test_idx in kf.split(selected_features, class_labels):
            X_train, X_test = selected_features[train_idx], selected_features[test_idx]
            y_train, y_test = class_labels[train_idx], class_labels[test_idx]

            model = model_template.__class__(**model_template.get_params())
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            f1_scores.append(f1_score(y_test, y_pred))
            roc_aucs.append(roc_auc_score(y_test, y_proba))

        results.append({
            "Features": ", ".join(feature_names_combination),
            "ROC-AUC": np.mean(roc_aucs),
            "F1-Score": np.mean(f1_scores)
        })

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    results_df.to_excel(result_file, index=False)
    print(f"Saved results to {result_file}")

    top_5 = results_df.head(5)

    for idx, row in top_5.iterrows():
        names = row["Features"].split(", ")
        indices = [list(feature_names).index(f) for f in names]
        selected_features = features[:, indices]
        shap_values_list = []

        for train_idx, test_idx in kf.split(selected_features, class_labels):
            X_train, X_test = selected_features[train_idx], selected_features[test_idx]
            y_train, y_test = class_labels[train_idx], class_labels[test_idx]

            model = model_template.__class__(**model_template.get_params())
            model.fit(X_train, y_train)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap_values_list.append(shap_values)

        all_shap = np.concatenate(shap_values_list, axis=0)

        plt.figure()
        shap.summary_plot(all_shap, features=selected_features, feature_names=names, show=False)
        plt.savefig(pics_folder / f"shap_plot_{idx + 1}.png", bbox_inches="tight")
        plt.close()

    print(f"Saved SHAP plots to {pics_folder}")

end_time = time.time()
print(f"\nAll processes completed in {end_time - start_time:.2f} seconds.")
