import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import ast

warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent / "data"
results_path = Path(__file__).parent.parent / "results" / "biomarker_analysis"
input_path = base_path / "pruned_preprocessed_data.xlsx"
f1_results_path = results_path / "f1_ranking" / "biomarker_model_results_f1_ranking.xlsx"
recall_results_path = results_path / "recall_ranking" / "biomarker_model_results_recall_ranking.xlsx"

plots_folder = results_path / "decision_boundary_plots"
os.makedirs(plots_folder, exist_ok=True)

data = pd.read_excel(input_path, header=None)
f1_results = pd.read_excel(f1_results_path)
recall_results = pd.read_excel(recall_results_path)

class_labels = data.iloc[0, 1:].values.astype(int)
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
X_scaled = scaler.fit_transform(X)

model_configs = {
    'SVM': {
        'model_class': SVC,
        'use_scaled': True
    },
    'Random Forest': {
        'model_class': RandomForestClassifier,
        'use_scaled': False
    },
    'XGBoost': {
        'model_class': XGBClassifier,
        'use_scaled': False
    },
    'MLP': {
        'model_class': MLPClassifier,
        'use_scaled': True
    }
}

def parse_params(param_string):
    try:
        params_dict = ast.literal_eval(param_string)
        
        if 'use_label_encoder' in params_dict:
            params_dict['use_label_encoder'] = False
        if 'eval_metric' in params_dict:
            params_dict['eval_metric'] = 'logloss'
        
        if 'max_iter' in params_dict:
            params_dict['max_iter'] = 2000
            
        return params_dict
    except Exception as e:
        print(f"Error parsing parameters: {param_string}")
        print(f"Error: {e}")
        return {}

def create_decision_boundary_plot(model, X_data, y_data, title, filename, optimization_type):
    plt.figure(figsize=(12, 10))
    
    h = 0.02
    x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
    y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    try:
        Z = model.predict_proba(mesh_points)[:, 1]
    except:
        try:
            Z = model.decision_function(mesh_points)
            Z = (Z - Z.min()) / (Z.max() - Z.min())
        except:
            Z = model.predict(mesh_points).astype(float)
    
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Prediction Probability (Class 1)')
    
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    colors = ['blue', 'red']
    labels = ['Non-Sepsis (Class 0)', 'Sepsis (Class 1)']
    
    for i, color, label in zip([0, 1], colors, labels):
        idx = y_data == i
        plt.scatter(X_data[idx, 0], X_data[idx, 1], 
                   c=color, s=60, alpha=0.8, edgecolors='black', 
                   linewidth=1, label=label)
    
    plt.xlabel(f'{target_biomarkers[0]} Expression', fontsize=12)
    plt.ylabel(f'{target_biomarkers[1]} Expression', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    textstr = f'Optimization: {optimization_type}\nDecision Boundary (dashed line at 0.5 probability)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def train_and_plot_model(model_name, params, optimization_type, results_row):
    print(f"\nProcessing {model_name} ({optimization_type} optimized)...")
    
    model_config = model_configs[model_name]
    X_data = X_scaled if model_config['use_scaled'] else X
    
    parsed_params = parse_params(params)
    if not parsed_params:
        print(f"Skipping {model_name} due to parameter parsing error")
        return
    
    try:
        model = model_config['model_class'](**parsed_params, random_state=42)
        model.fit(X_data, y)
        
        y_pred = model.predict(X_data)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        
        print(f"  Trained model metrics:")
        print(f"    Recall: {recall:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        
        expected_recall = results_row['Recall_Mean']
        expected_f1 = results_row['F1_Score_Mean']
        expected_precision = results_row['Precision_Mean']
        
        print(f"  Expected metrics from CV:")
        print(f"    Recall: {expected_recall:.4f}")
        print(f"    Precision: {expected_precision:.4f}")
        print(f"    F1-Score: {expected_f1:.4f}")
        
        title = (f'{model_name} Decision Boundary ({optimization_type.upper()}-Optimized)\n'
                f'Recall: {expected_recall:.3f} | Precision: {expected_precision:.3f} | F1: {expected_f1:.3f}')
        
        filename = plots_folder / f"decision_boundary_{model_name.lower().replace(' ', '_')}_{optimization_type.lower()}_optimized.png"
        
        create_decision_boundary_plot(model, X_data, y, title, filename, optimization_type)
        print(f"  Decision boundary plot saved: {filename}")
        
    except Exception as e:
        print(f"  Error training {model_name}: {e}")

print("\n" + "="*80)
print("CREATING DECISION BOUNDARY PLOTS FOR F1-OPTIMIZED MODELS")
print("="*80)

for idx, row in f1_results.iterrows():
    model_name = row['Model']
    params = row['Best_Parameters']
    train_and_plot_model(model_name, params, 'F1', row)

print("\n" + "="*80)
print("CREATING DECISION BOUNDARY PLOTS FOR RECALL-OPTIMIZED MODELS")
print("="*80)

for idx, row in recall_results.iterrows():
    model_name = row['Model']
    params = row['Best_Parameters']
    train_and_plot_model(model_name, params, 'Recall', row)

print("\n" + "="*80)
print("CREATING COMPARISON PLOTS FOR BEST MODELS")
print("="*80)

best_f1_row = f1_results.iloc[0]
best_recall_row = recall_results.iloc[0]

print(f"Best F1 model: {best_f1_row['Model']} (F1: {best_f1_row['F1_Score_Mean']:.4f})")
print(f"Best Recall model: {best_recall_row['Model']} (Recall: {best_recall_row['Recall_Mean']:.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

def plot_decision_boundary_on_axis(ax, model, X_data, y_data, title, optimization_type):
    h = 0.02
    x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
    y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    try:
        Z = model.predict_proba(mesh_points)[:, 1]
    except:
        try:
            Z = model.decision_function(mesh_points)
            Z = (Z - Z.min()) / (Z.max() - Z.min())
        except:
            Z = model.predict(mesh_points).astype(float)
    
    Z = Z.reshape(xx.shape)
    
    contour = ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    colors = ['blue', 'red']
    labels = ['Non-Sepsis (Class 0)', 'Sepsis (Class 1)']
    
    for i, color, label in zip([0, 1], colors, labels):
        idx = y_data == i
        ax.scatter(X_data[idx, 0], X_data[idx, 1], 
                  c=color, s=60, alpha=0.8, edgecolors='black', 
                  linewidth=1, label=label)
    
    ax.set_xlabel(f'{target_biomarkers[0]} Expression', fontsize=12)
    ax.set_ylabel(f'{target_biomarkers[1]} Expression', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return contour

try:
    f1_model_config = model_configs[best_f1_row['Model']]
    f1_X_data = X_scaled if f1_model_config['use_scaled'] else X
    f1_params = parse_params(best_f1_row['Best_Parameters'])
    f1_model = f1_model_config['model_class'](**f1_params, random_state=42)
    f1_model.fit(f1_X_data, y)
    
    f1_title = (f"Best F1 Model: {best_f1_row['Model']}\n"
               f"F1: {best_f1_row['F1_Score_Mean']:.3f} | "
               f"Recall: {best_f1_row['Recall_Mean']:.3f} | "
               f"Precision: {best_f1_row['Precision_Mean']:.3f}")
    
    contour1 = plot_decision_boundary_on_axis(ax1, f1_model, f1_X_data, y, f1_title, 'F1')
    
except Exception as e:
    print(f"Error plotting best F1 model: {e}")

try:
    recall_model_config = model_configs[best_recall_row['Model']]
    recall_X_data = X_scaled if recall_model_config['use_scaled'] else X
    recall_params = parse_params(best_recall_row['Best_Parameters'])
    recall_model = recall_model_config['model_class'](**recall_params, random_state=42)
    recall_model.fit(recall_X_data, y)
    
    recall_title = (f"Best Recall Model: {best_recall_row['Model']}\n"
                   f"Recall: {best_recall_row['Recall_Mean']:.3f} | "
                   f"F1: {best_recall_row['F1_Score_Mean']:.3f} | "
                   f"Precision: {best_recall_row['Precision_Mean']:.3f}")
    
    contour2 = plot_decision_boundary_on_axis(ax2, recall_model, recall_X_data, y, recall_title, 'Recall')
    
except Exception as e:
    print(f"Error plotting best Recall model: {e}")

cbar = fig.colorbar(contour1, ax=[ax1, ax2], orientation='horizontal', pad=0.1, aspect=50)
cbar.set_label('Prediction Probability (Class 1)', fontsize=12)

plt.suptitle('Comparison: Best F1-Optimized vs Best Recall-Optimized Models', 
             fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()

comparison_filename = plots_folder / "best_models_comparison.png"
plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"Comparison plot saved: {comparison_filename}")

print("\n" + "="*80)
print("DECISION BOUNDARY VISUALIZATION COMPLETE!")
print("="*80)
print(f"All plots saved to: {plots_folder}")
print(f"Individual model plots: decision_boundary_[model]_[optimization]_optimized.png")
print(f"Comparison plot: best_models_comparison.png")
print("\nDecision boundary interpretation:")
print("- Blue regions: Model predicts Non-Sepsis (Class 0)")
print("- Red regions: Model predicts Sepsis (Class 1)")
print("- Black dashed line: Decision boundary (0.5 probability threshold)")
print("- Blue points: Actual Non-Sepsis samples")
print("- Red points: Actual Sepsis samples")
print("="*80)
