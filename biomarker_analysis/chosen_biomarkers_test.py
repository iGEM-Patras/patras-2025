"""Four classifiers on the selected hsa-miR-150-5p / hsa-miR-486-5p pair.

SUPERSEDED FOR PERFORMANCE ESTIMATION. This script standardises the full matrix
once, before the cross-validation loop (`scaler.fit_transform(X)` below), so the
scaler is fitted on data that later serves as held-out folds. The resulting
metrics are therefore mildly optimistic and are NOT the performance estimates
reported in the manuscript.

The leak-free estimates come from `leakfree_cv_analysis.py`, where scaling and
feature selection sit inside an sklearn Pipeline and are refitted within every
fold, with nested cross-validation for the hyper-parameter search.

ITS SHAP PLOTS ARE ALSO SUPERSEDED. The summary plots it draws pass the per-fold
SHAP values together with the feature matrix in its original row order, so row i
of the attribution matrix does not correspond to row i of the features and the
colour encoding is scrambled. Manuscript Figure 2 is produced by `shap_figure.py`,
which concatenates the held-out feature rows in the same fold order.

This script is retained because it produces the model comparison table and because
the decision-boundary panels of Figure 1 are drawn from its output. It is deposited
unchanged so that those remain reproducible; its SHAP output is not used in the
manuscript.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent / "data"
results_path = Path(__file__).parent.parent / "results" / "biomarker_analysis"
input_path = base_path / "pruned_preprocessed_data.xlsx"

f1_folder = results_path / "f1_ranking"
recall_folder = results_path / "recall_ranking"
os.makedirs(f1_folder, exist_ok=True)
os.makedirs(recall_folder, exist_ok=True)

f1_output_path = f1_folder / "biomarker_model_results_f1_ranking.xlsx"
recall_output_path = recall_folder / "biomarker_model_results_recall_ranking.xlsx"
f1_plots_folder = f1_folder / "shap_plots"
recall_plots_folder = recall_folder / "shap_plots"

os.makedirs(f1_plots_folder, exist_ok=True)
os.makedirs(recall_plots_folder, exist_ok=True)

print("Loading data...")
data = pd.read_excel(input_path, header=None)

class_labels = data.iloc[0, 1:].values.astype(int)
print(f"Class distribution: {np.bincount(class_labels)}")

feature_names = data.iloc[1:, 0].values

target_biomarkers = ['hsa-miR-150-5p', 'hsa-miR-486-5p']
biomarker_indices = []
for biomarker in target_biomarkers:
    if biomarker in feature_names:
        idx = list(feature_names).index(biomarker)
        biomarker_indices.append(idx + 1)  
        print(f"Found {biomarker} at row {idx + 1}")
    else:
        print(f"Warning: {biomarker} not found in dataset")

if len(biomarker_indices) != 2:
    raise ValueError("Both target biomarkers must be present in the dataset")

selected_rows = [0] + biomarker_indices
subset_data = data.iloc[selected_rows, :]

X = subset_data.iloc[1:, 1:].T.values
y = class_labels

print(f"Dataset shape: {X.shape}")
print(f"Features: {target_biomarkers}")
print(f"Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")

scaler = StandardScaler()
# NOTE: fitted on the full matrix, before the CV split below - see the module
# docstring. Leak-free estimates are in leakfree_cv_analysis.py.
X_scaled = scaler.fit_transform(X)

models = {
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'degree': [2, 3, 4],
            'coef0': [0.0, 1.0]
        },
        'use_scaled': True
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'criterion': ['gini', 'entropy']
        },
        'use_scaled': False
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [0, 0.1, 1.0]
        },
        'use_scaled': False
    },
    'MLP': {
        'model': MLPClassifier(random_state=42, max_iter=2000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (150, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'solver': ['adam', 'lbfgs']
        },
        'use_scaled': True
    }
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
recall_scorer = make_scorer(recall_score)
f1_scorer = make_scorer(f1_score)

results_recall_optimized = []
results_f1_optimized = []

print("\nTraining models with REDUCED Grid Search...")
print("Will run TWO optimization rounds:")
print("1. Models optimized for RECALL (then ranked by recall)")
print("2. Models optimized for F1-SCORE (then ranked by F1)")
print("Estimated runtime: 30-60 minutes with reduced parameter combinations")
start_time = time.time()

total_all_combinations = 0
for model_name, model_config in models.items():
    model_combinations = 1
    for param_name, param_values in model_config['params'].items():
        model_combinations *= len(param_values)
    total_all_combinations += model_combinations

print(f"TOTAL PARAMETER COMBINATIONS PER OPTIMIZATION ROUND: {total_all_combinations:,}")
print(f"TOTAL PARAMETER COMBINATIONS FOR BOTH ROUNDS: {total_all_combinations * 2:,}")
print("="*80)

# First optimization round: RECALL-based optimization
print("\n" + "="*80)
print("OPTIMIZATION ROUND 1: OPTIMIZING FOR RECALL")
print("="*80)

completed_combinations = 0
model_counter = 0

for model_name, model_config in models.items():
    model_counter += 1
    model_start_time = time.time()
    print(f"\n[RECALL OPT - MODEL {model_counter}/4] Training {model_name}...")
    
    model_combinations = 1
    for param_name, param_values in model_config['params'].items():
        model_combinations *= len(param_values)
    print(f"   Parameter combinations for {model_name}: {model_combinations:,}")
    print(f"   Round 1 progress: {completed_combinations:,}/{total_all_combinations:,} combinations completed")
    
    X_train = X_scaled if model_config['use_scaled'] else X
    
    class VerboseGridSearch:
        def __init__(self, estimator, param_grid, scoring, cv, n_jobs, optimization_metric):
            self.estimator = estimator
            self.param_grid = param_grid
            self.scoring = scoring
            self.cv = cv
            self.n_jobs = n_jobs
            self.completed = 0
            self.total = model_combinations
            self.optimization_metric = optimization_metric
            
        def fit(self, X, y):
            from sklearn.model_selection import ParameterGrid
            from sklearn.base import clone
            
            param_combinations = list(ParameterGrid(self.param_grid))
            self.total = len(param_combinations)
            
            best_score = -np.inf
            best_params = None
            best_estimator = None
            
            print(f"   Starting {model_name} grid search (optimizing for {self.optimization_metric}): {self.total:,} combinations to test...")
            print(f"   Progress: 0/{self.total} (0.0%)")
            
            for i, params in enumerate(param_combinations):
                model = clone(self.estimator)
                model.set_params(**params)
                
                scores = []
                for train_idx, val_idx in self.cv.split(X, y):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    
                    if self.optimization_metric == "RECALL":
                        score = recall_score(y_val_fold, y_pred)
                    elif self.optimization_metric == "F1":
                        score = f1_score(y_val_fold, y_pred)
                    
                    scores.append(score)
                
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    best_estimator = clone(self.estimator)
                    best_estimator.set_params(**params)
                
                self.completed = i + 1
                progress_pct = (self.completed / self.total) * 100
                
                if (self.completed % 50 == 0 or 
                    self.completed in [1, 10, 25, 50, 100, 250, 500, 1000] or
                    self.completed == self.total or
                    progress_pct in [25, 50, 75, 90, 95]):
                    
                    elapsed = time.time() - model_start_time
                    if self.completed > 1:
                        time_per_combo = elapsed / self.completed
                        remaining_combos = self.total - self.completed
                        eta_seconds = time_per_combo * remaining_combos
                        eta_minutes = eta_seconds / 60
                        eta_str = f", ETA: {eta_minutes:.1f}m" if eta_minutes > 1 else f", ETA: {eta_seconds:.0f}s"
                    else:
                        eta_str = ""
                    
                    print(f"   Progress: {self.completed:,}/{self.total:,} ({progress_pct:.1f}%) - "
                          f"Best {self.optimization_metric} so far: {best_score:.4f}{eta_str}")
            
            class GridSearchResult:
                def __init__(self, best_estimator, best_params, best_score):
                    self.best_estimator_ = best_estimator
                    self.best_params_ = best_params
                    self.best_score_ = best_score
            
            return GridSearchResult(best_estimator, best_params, best_score)
    
    verbose_grid_search = VerboseGridSearch(
        estimator=model_config['model'],
        param_grid=model_config['params'],
        scoring=recall_scorer,
        cv=cv,
        n_jobs=-1,
        optimization_metric="RECALL"
    )
    
    grid_result = verbose_grid_search.fit(X_train, y)
    
    completed_combinations += model_combinations
    
    model_end_time = time.time()
    print(f"   ✓ {model_name} completed in {model_end_time - model_start_time:.2f} seconds")
    print(f"   ✓ Best {model_name} recall: {grid_result.best_score_:.4f}")
    print(f"   ✓ Round 1 progress: {completed_combinations:,}/{total_all_combinations:,} combinations completed")
    print("   " + "="*60)
    
    best_model = grid_result.best_estimator_
    
    # Evaluate the recall-optimized model on all metrics
    recall_scores = []
    precision_scores = []
    f1_scores = []
    
    for train_idx, test_idx in cv.split(X_train, y):
        X_train_fold, X_test_fold = X_train[train_idx], X_train[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        fold_model = model_config['model'].__class__(**best_model.get_params())
        fold_model.fit(X_train_fold, y_train_fold)
        
        y_pred = fold_model.predict(X_test_fold)

        recall_scores.append(recall_score(y_test_fold, y_pred))
        precision_scores.append(precision_score(y_test_fold, y_pred))
        f1_scores.append(f1_score(y_test_fold, y_pred))
    
    results_recall_optimized.append({
        'Model': model_name,
        'Best_Parameters': str(grid_result.best_params_),
        'Recall_Mean': np.mean(recall_scores),
        'Recall_Std': np.std(recall_scores),
        'Precision_Mean': np.mean(precision_scores),
        'Precision_Std': np.std(precision_scores),
        'F1_Score_Mean': np.mean(f1_scores),
        'F1_Score_Std': np.std(f1_scores),
        'Best_Model': best_model,
        'Use_Scaled_Data': model_config['use_scaled'],
        'Optimization_Metric': 'Recall'
    })
    
    print(f"{model_name} (Recall-optimized) - Recall: {np.mean(recall_scores):.4f} (±{np.std(recall_scores):.4f})")

# Second optimization round: F1-based optimization
print("\n" + "="*80)
print("OPTIMIZATION ROUND 2: OPTIMIZING FOR F1-SCORE")
print("="*80)

completed_combinations = 0
model_counter = 0

for model_name, model_config in models.items():
    model_counter += 1
    model_start_time = time.time()
    print(f"\n[F1 OPT - MODEL {model_counter}/4] Training {model_name}...")
    
    model_combinations = 1
    for param_name, param_values in model_config['params'].items():
        model_combinations *= len(param_values)
    print(f"   Parameter combinations for {model_name}: {model_combinations:,}")
    print(f"   Round 2 progress: {completed_combinations:,}/{total_all_combinations:,} combinations completed")
    
    X_train = X_scaled if model_config['use_scaled'] else X
    
    verbose_grid_search = VerboseGridSearch(
        estimator=model_config['model'],
        param_grid=model_config['params'],
        scoring=f1_scorer,
        cv=cv,
        n_jobs=-1,
        optimization_metric="F1"
    )
    
    grid_result = verbose_grid_search.fit(X_train, y)
    
    completed_combinations += model_combinations
    
    model_end_time = time.time()
    print(f"   ✓ {model_name} completed in {model_end_time - model_start_time:.2f} seconds")
    print(f"   ✓ Best {model_name} F1: {grid_result.best_score_:.4f}")
    print(f"   ✓ Round 2 progress: {completed_combinations:,}/{total_all_combinations:,} combinations completed")
    print("   " + "="*60)
    
    best_model = grid_result.best_estimator_
    
    # Evaluate the F1-optimized model on all metrics
    recall_scores = []
    precision_scores = []
    f1_scores = []
    
    for train_idx, test_idx in cv.split(X_train, y):
        X_train_fold, X_test_fold = X_train[train_idx], X_train[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        fold_model = model_config['model'].__class__(**best_model.get_params())
        fold_model.fit(X_train_fold, y_train_fold)
        
        y_pred = fold_model.predict(X_test_fold)

        recall_scores.append(recall_score(y_test_fold, y_pred))
        precision_scores.append(precision_score(y_test_fold, y_pred))
        f1_scores.append(f1_score(y_test_fold, y_pred))
    
    results_f1_optimized.append({
        'Model': model_name,
        'Best_Parameters': str(grid_result.best_params_),
        'Recall_Mean': np.mean(recall_scores),
        'Recall_Std': np.std(recall_scores),
        'Precision_Mean': np.mean(precision_scores),
        'Precision_Std': np.std(precision_scores),
        'F1_Score_Mean': np.mean(f1_scores),
        'F1_Score_Std': np.std(f1_scores),
        'Best_Model': best_model,
        'Use_Scaled_Data': model_config['use_scaled'],
        'Optimization_Metric': 'F1'
    })
    
    print(f"{model_name} (F1-optimized) - F1: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")

# Sort results by their respective optimization metrics
results_recall = sorted(results_recall_optimized, key=lambda x: x['Recall_Mean'], reverse=True)
results_f1 = sorted(results_f1_optimized, key=lambda x: x['F1_Score_Mean'], reverse=True)

def create_results_df(sorted_results, ranking_type):
    return pd.DataFrame([
        {
            'Rank': i + 1,
            'Model': result['Model'],
            'Optimization_Metric': result['Optimization_Metric'],
            'Ranking_Metric': ranking_type,
            'Recall_Mean': result['Recall_Mean'],
            'Recall_Std': result['Recall_Std'],
            'Precision_Mean': result['Precision_Mean'],
            'Precision_Std': result['Precision_Std'],
            'F1_Score_Mean': result['F1_Score_Mean'],
            'F1_Score_Std': result['F1_Score_Std'],
            'Best_Parameters': result['Best_Parameters']
        }
        for i, result in enumerate(sorted_results)
    ])

recall_results_df = create_results_df(results_recall, "Recall")
f1_results_df = create_results_df(results_f1, "F1_Score")

recall_results_df.to_excel(recall_output_path, index=False)
f1_results_df.to_excel(f1_output_path, index=False)

print(f"\nRecall-optimized results (ranked by recall) saved to {recall_output_path}")
print(f"F1-optimized results (ranked by F1) saved to {f1_output_path}")

print("\nModel Ranking - RECALL-OPTIMIZED models ranked by RECALL:")
print("=" * 80)
for i, result in enumerate(results_recall, 1):
    print(f"{i}. {result['Model']} (optimized for {result['Optimization_Metric']})")
    print(f"   Recall: {result['Recall_Mean']:.4f} ± {result['Recall_Std']:.4f}")
    print(f"   Precision: {result['Precision_Mean']:.4f} ± {result['Precision_Std']:.4f}")
    print(f"   F1-Score: {result['F1_Score_Mean']:.4f} ± {result['F1_Score_Std']:.4f}")
    print(f"   Best Parameters: {result['Best_Parameters']}")
    print()

print("\nModel Ranking - F1-OPTIMIZED models ranked by F1 SCORE:")
print("=" * 80)
for i, result in enumerate(results_f1, 1):
    print(f"{i}. {result['Model']} (optimized for {result['Optimization_Metric']})")
    print(f"   F1-Score: {result['F1_Score_Mean']:.4f} ± {result['F1_Score_Std']:.4f}")
    print(f"   Recall: {result['Recall_Mean']:.4f} ± {result['Recall_Std']:.4f}")
    print(f"   Precision: {result['Precision_Mean']:.4f} ± {result['Precision_Std']:.4f}")
    print(f"   Best Parameters: {result['Best_Parameters']}")
    print()

def generate_shap_analysis(sorted_results, plots_folder, ranking_name):
    print(f"\nGenerating SHAP values and plots for {ranking_name} ranking...")
    
    for i, result in enumerate(sorted_results):
        model_name = result['Model']
        model = result['Best_Model']
        use_scaled = result['Use_Scaled_Data']
        
        print(f"\nGenerating SHAP plot for {model_name} ({ranking_name} ranking - rank {i+1})...")
        
        X_for_shap = X_scaled if use_scaled else X
        
        try:
            if model_name in ['Random Forest', 'XGBoost']:
                shap_values_list = []
                
                for train_idx, test_idx in cv.split(X_for_shap, y):
                    X_train_fold, X_test_fold = X_for_shap[train_idx], X_for_shap[test_idx]
                    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                    
                    fold_model = model.__class__(**model.get_params())
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    explainer = shap.TreeExplainer(fold_model)
                    shap_values_fold = explainer.shap_values(X_test_fold)
                    
                    if isinstance(shap_values_fold, list):
                        shap_values_list.append(shap_values_fold[1])
                        print(f"     Fold SHAP values (list format) shape: {shap_values_fold[1].shape}")
                    elif len(shap_values_fold.shape) == 3:
                        shap_values_list.append(shap_values_fold[:, :, 1])
                        print(f"     Fold SHAP values (3D format) shape: {shap_values_fold[:, :, 1].shape}")
                    else:
                        shap_values_list.append(shap_values_fold)
                        print(f"     Fold SHAP values (2D format) shape: {shap_values_fold.shape}")
                
                all_shap_values = np.concatenate(shap_values_list, axis=0)
                
                mean_shap_values = np.mean(np.abs(all_shap_values), axis=0)
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(all_shap_values, 
                                features=X_for_shap, 
                                feature_names=target_biomarkers, 
                                show=False)
                
                plt.title(f'SHAP Summary Plot - {model_name} ({ranking_name} Rank #{i+1})\n'
                         f'Recall: {result["Recall_Mean"]:.4f} ± {result["Recall_Std"]:.4f}, '
                         f'F1: {result["F1_Score_Mean"]:.4f} ± {result["F1_Score_Std"]:.4f}')
                plt.tight_layout()
                
                plot_filename = plots_folder / f"shap_{model_name.lower().replace(' ', '_')}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   SHAP plot saved: {plot_filename}")
                print(f"   Feature importance - {target_biomarkers[0]}: {mean_shap_values[0]:.6f}")
                print(f"   Feature importance - {target_biomarkers[1]}: {mean_shap_values[1]:.6f}")
                
                plt.figure(figsize=(10, 6))
                
                model.fit(X_for_shap, y)
                feature_importance = model.feature_importances_
                
                plt.bar(target_biomarkers, feature_importance)
                plt.title(f'Feature Importances - {model_name} ({ranking_name} Rank #{i+1})\n'
                         f'Recall: {result["Recall_Mean"]:.4f} ± {result["Recall_Std"]:.4f}, '
                         f'F1: {result["F1_Score_Mean"]:.4f} ± {result["F1_Score_Std"]:.4f}')
                plt.ylabel('Importance')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                importance_filename = plots_folder / f"importance_{model_name.lower().replace(' ', '_')}.png"
                plt.savefig(importance_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   Feature importance plot saved: {importance_filename}")
                print(f"   Built-in importance - {target_biomarkers[0]}: {feature_importance[0]:.6f}")
                print(f"   Built-in importance - {target_biomarkers[1]}: {feature_importance[1]:.6f}")
                
            else:
                sample_size = min(100, len(X_for_shap))
                sample_indices = np.random.choice(len(X_for_shap), sample_size, replace=False)
                X_sample = X_for_shap[sample_indices]
                y_sample = y[sample_indices]
                
                model.fit(X_for_shap, y)
                
                def model_predict(X):
                    probs = model.predict_proba(X)
                    return probs[:, 1]
                
                explainer = shap.KernelExplainer(model_predict, X_sample[:20])
                shap_values = explainer.shap_values(X_sample)
                
                mean_shap_values = np.mean(np.abs(shap_values), axis=0)
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, 
                                features=X_sample, 
                                feature_names=target_biomarkers, 
                                show=False)
                
                plt.title(f'SHAP Summary Plot - {model_name} ({ranking_name} Rank #{i+1})\n'
                         f'Recall: {result["Recall_Mean"]:.4f} ± {result["Recall_Std"]:.4f}, '
                         f'F1: {result["F1_Score_Mean"]:.4f} ± {result["F1_Score_Std"]:.4f}')
                plt.tight_layout()
                
                plot_filename = plots_folder / f"shap_{model_name.lower().replace(' ', '_')}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   SHAP plot saved: {plot_filename}")
                print(f"   Feature importance - {target_biomarkers[0]}: {mean_shap_values[0]:.6f}")
                print(f"   Feature importance - {target_biomarkers[1]}: {mean_shap_values[1]:.6f}")
            
        except Exception as e:
            print(f"   SHAP analysis failed for {model_name}: {e}")
            
            plt.figure(figsize=(10, 6))
            
            model.fit(X_for_shap, y)
            
            if hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_[0])
                plot_title = f'Feature Coefficients (Absolute) - {model_name}'
            elif hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                plot_title = f'Feature Importances - {model_name}'
            else:
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(model, X_for_shap, y, n_repeats=10, random_state=42)
                feature_importance = perm_importance.importances_mean
                plot_title = f'Permutation Feature Importance - {model_name}'
            
            plt.bar(target_biomarkers, feature_importance)
            plt.title(f'{plot_title} ({ranking_name} Rank #{i+1})\n'
                     f'Recall: {result["Recall_Mean"]:.4f} ± {result["Recall_Std"]:.4f}, '
                     f'F1: {result["F1_Score_Mean"]:.4f} ± {result["F1_Score_Std"]:.4f}')
            plt.ylabel('Importance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_filename = plots_folder / f"importance_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   Feature importance - {target_biomarkers[0]}: {feature_importance[0]:.6f}")
            print(f"   Feature importance - {target_biomarkers[1]}: {feature_importance[1]:.6f}")

generate_shap_analysis(results_recall, recall_plots_folder, "Recall")
generate_shap_analysis(results_f1, f1_plots_folder, "F1")

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

print("\nAnalysis complete!")
print(f"Best RECALL-OPTIMIZED model (ranked by recall): {results_recall[0]['Model']}")
print(f"  - Optimized for: {results_recall[0]['Optimization_Metric']}")
print(f"  - Best recall score: {results_recall[0]['Recall_Mean']:.4f} ± {results_recall[0]['Recall_Std']:.4f}")
print(f"  - F1 score: {results_recall[0]['F1_Score_Mean']:.4f} ± {results_recall[0]['F1_Score_Std']:.4f}")
print(f"Best F1-OPTIMIZED model (ranked by F1): {results_f1[0]['Model']}")
print(f"  - Optimized for: {results_f1[0]['Optimization_Metric']}")
print(f"  - Best F1 score: {results_f1[0]['F1_Score_Mean']:.4f} ± {results_f1[0]['F1_Score_Std']:.4f}")
print(f"  - Recall score: {results_f1[0]['Recall_Mean']:.4f} ± {results_f1[0]['Recall_Std']:.4f}")

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Dataset: {len(y)} samples with 2 biomarkers")
print(f"Target biomarkers: {', '.join(target_biomarkers)}")
print(f"Class distribution: {np.sum(y == 0)} non-sepsis, {np.sum(y == 1)} sepsis")
print(f"\nRECALL-OPTIMIZED MODELS (ranked by recall performance):")
for i, result in enumerate(results_recall, 1):
    print(f"{i}. {result['Model']}: Recall={result['Recall_Mean']:.4f}, "
          f"Precision={result['Precision_Mean']:.4f}, F1={result['F1_Score_Mean']:.4f}")
print(f"\nF1-OPTIMIZED MODELS (ranked by F1 performance):")
for i, result in enumerate(results_f1, 1):
    print(f"{i}. {result['Model']}: F1={result['F1_Score_Mean']:.4f}, "
          f"Recall={result['Recall_Mean']:.4f}, Precision={result['Precision_Mean']:.4f}")
print(f"\nRecall-optimized results saved to: {recall_output_path}")
print(f"F1-optimized results saved to: {f1_output_path}")
print(f"Recall-optimized SHAP plots saved to: {recall_plots_folder}")
print(f"F1-optimized SHAP plots saved to: {f1_plots_folder}")
print("="*80)
