import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_features):
        sns.boxplot(x='readmitted_lt_30', y=col, data=df,
                    ax=axes[i], showfliers=False)
        axes[i].set_title(f'{col} vs Target')

    plt.tight_layout()
    plt.savefig("figures/EDA/numeric_boxplots.png")
    plt.close()
    print("Saved: figures/EDA/numeric_boxplots.png")

    print("--- EDA Plots Saved Successfully ---\n")


# ============================================================
#                M O D E L   E V A L U A T I O N
# ============================================================
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n--- Results for {model_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(classification_report(y_test, y_pred))

    return {
        'model_name': model_name,
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc,
        'y_pred': y_pred,
        'model_obj': model
    }


# ============================================================
#               R E S U L T   P L O T S
# ============================================================
def plot_results(results_list):
    results_df = pd.DataFrame(results_list).set_index('model_name')

    # Comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    results_df.sort_values('f1_score', ascending=False, inplace=True)
    sns.barplot(x=results_df.f1_score, y=results_df.index, ax=axes[0])
    axes[0].set_title('Model F1-Score Comparison')

    results_df.sort_values('roc_auc', ascending=False, inplace=True)
    sns.barplot(x=results_df.roc_auc, y=results_df.index, ax=axes[1])
    axes[1].set_title('Model ROC-AUC Comparison')

    plt.tight_layout()
    plt.savefig("figures/Results/model_comparison.png")
    plt.close()

    print("Saved: figures/Results/model_comparison.png")
    print("\nSummary Results:")
    print(results_df[['accuracy', 'f1_score', 'roc_auc']])


# ============================================================
#            F E A T U R E   I M P O R T A N C E
# ============================================================
def plot_feature_importance(best_model_pipeline, feature_names):
    model = best_model_pipeline.named_steps['model']

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        feat_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_df.head(25))
        plt.title('Top 25 Feature Importances')
        plt.tight_layout()

        plt.savefig("figures/Results/feature_importance.png")
        plt.close()

        print("Saved: figures/Results/feature_importance.png")

    else:
        print("This model does not support feature importance.")
