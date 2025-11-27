import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# Create figures directory if not exists
os.makedirs("figures", exist_ok=True)
os.makedirs("figures/EDA", exist_ok=True)
os.makedirs("figures/Results", exist_ok=True)

plt.style.use('ggplot')
sns.set_palette('colorblind')

# ============================================================
#                     L O G G I N G
# ============================================================
def log_experiment(model_name, metrics, note=""):
    """
    Logs the experiment results to a CSV file.
    """
    file_exists = os.path.isfile('experiments.csv')
    with open('experiments.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file is new
        if not file_exists:
            writer.writerow(['Timestamp', 'Model', 'Accuracy', 'F1_Score', 'ROC_AUC', 'Note'])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            round(metrics['accuracy'], 4),
            round(metrics['f1_score'], 4),
            round(metrics['roc_auc'], 4),
            note
        ])
    print(f"   -> Logged {model_name} to experiments.csv")

# ============================================================
#                      E D A   P L O T S
# ============================================================
def plot_eda(df):
    print("--- Generating EDA Plots (saving instead of showing) ---")

    # --- Target Distribution ---
    plt.figure(figsize=(6, 4))
    sns.countplot(x='readmitted_lt_30', data=df)
    plt.title('Target Distribution')
    plt.tight_layout()
    plt.savefig("figures/EDA/target_distribution.png")
    plt.close()
    print("Saved: figures/EDA/target_distribution.png")

    # --- Numeric Boxplots ---
    numeric_features = [
        'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications'
    ]
    
    # Check if columns exist before plotting
    numeric_features = [c for c in numeric_features if c in df.columns]

    if numeric_features:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for i, col in enumerate(numeric_features):
            sns.boxplot(x='readmitted_lt_30', y=col, data=df, ax=axes[i])
            axes[i].set_title(f'{col} vs Target')
        plt.tight_layout()
        plt.savefig("figures/EDA/numeric_boxplots.png")
        plt.close()
        print("Saved: figures/EDA/numeric_boxplots.png")
    
    print("--- EDA Plots Saved Successfully ---")

# ============================================================
#                 E V A L U A T I O N
# ============================================================
def evaluate_model(model_pipeline, X_test, y_test, model_name):
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1] if hasattr(model_pipeline, "predict_proba") else y_pred
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary') # Binary since we care about class 1
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n--- Results for {model_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc
    }
    
    # --- LOGGING CALL ADDED HERE ---
    log_experiment(model_name, metrics, note="Phase 3 Run - SMOTE")
    
    return {
        'model_name': model_name,
        'model_obj': model_pipeline,
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc
    }

def plot_results(results_list):
    results_df = pd.DataFrame(results_list)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot F1
    results_df.sort_values('f1_score', ascending=False, inplace=True)
    sns.barplot(x=results_df.f1_score, y=results_df.model_name, ax=axes[0], palette='viridis')
    axes[0].set_title('Model F1-Score Comparison')
    
    # Plot ROC-AUC
    results_df.sort_values('roc_auc', ascending=False, inplace=True)
    sns.barplot(x=results_df.roc_auc, y=results_df.model_name, ax=axes[1], palette='viridis')
    axes[1].set_title('Model ROC-AUC Comparison')
    
    plt.tight_layout()
    plt.savefig("figures/Results/model_comparison.png")
    plt.close()
    
    print("Saved: figures/Results/model_comparison.png")
    print("\nSummary Results:")
    print(results_df[['model_name', 'accuracy', 'f1_score', 'roc_auc']])


# ============================================================
#            F E A T U R E   I M P O R T A N C E
# ============================================================
def plot_feature_importance(best_model_pipeline, feature_names):
    model = best_model_pipeline.named_steps['model']
    importances = None
    
    # Check for Tree-based feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    
    # Check for Linear Model coefficients (Logistic Regression)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        
    if importances is not None:
        # Create a dataframe for visualization
        # Ensure lengths match (sometimes encoders drop features)
        if len(importances) != len(feature_names):
            print(f"Warning: Feature names ({len(feature_names)}) and Importance length ({len(importances)}) mismatch.")
            return

        feat_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_df.head(20), palette='magma')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig("figures/Results/feature_importance.png")
        plt.close()
        print("Saved: figures/Results/feature_importance.png")
    else:
        print("This model does not support feature importance plotting.")