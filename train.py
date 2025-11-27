import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Custom modules
from src import preprocess
from src import feature_engineering
from src import evaluate

def main():
    # 1. Load and Clean
    df = preprocess.load_and_clean_data('data/diabetic_data.csv')
    
    # 2. Feature Engineering
    df = feature_engineering.apply_feature_engineering(df)
    
    # 3. Exploratory Data Analysis (Optional - show plots)
    evaluate.plot_eda(df)
    
    # 4. Prepare for Training
    num_feats, ord_feats, cat_feats, age_order = preprocess.get_feature_lists()
    preprocessor = preprocess.build_pipeline(num_feats, ord_feats, cat_feats, age_order)
    
    X = df.drop(columns=['readmitted_lt_30'])
    y = df['readmitted_lt_30']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Define Models
    
    # Default configuration for Random Forest
    rf_params = {
        'random_state': 42, 
        'class_weight': 'balanced', 
        'n_jobs': -1
    }

    # Check if 'best_params.json' exists (created by tune.py)
    if os.path.exists('best_params.json'):
        with open('best_params.json', 'r') as f:
            tuned_params = json.load(f)
            # Merge tuned params into the default config
            rf_params.update(tuned_params)
            print(f"\n--> LOADED TUNED PARAMS FOR RF: {tuned_params}")
    else:
        print("\n--> Using DEFAULT parameters for RF (Run tune.py to optimize)")

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        
        # We unpack the dictionary (**rf_params) to inject the settings
        'Random Forest': RandomForestClassifier(**rf_params),
        
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # 6. Train and Evaluate Loop
    results_list = []
    
    print("--- Starting Model Training ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use ImbPipeline to ensure SMOTE is only applied to the training folds
        pipe = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)), # Synthesizes new examples of the minority class
            ('model', model)
        ])
        
        pipe.fit(X_train, y_train)
        
        # Evaluate
        res = evaluate.evaluate_model(pipe, X_test, y_test, name)
        results_list.append(res)
        
    print("--- Model Training Complete ---")
        
    # 7. Final Visualizations
    evaluate.plot_results(results_list)
    
    # Feature Importance for the best model (by F1)
    best_result = max(results_list, key=lambda x: x['f1_score'])
    best_model = best_result['model_obj']
    
    # Helper to reconstruct feature names for the plot
    prep = best_model.named_steps['preprocessor']
    cat_feats_full = [c for c in X.columns if c in cat_feats or c == 'diag_1_group']
    cat_names = prep.named_transformers_['cat']['encoder'].get_feature_names_out(cat_feats_full)
    all_feature_names = num_feats + ord_feats + list(cat_names)
    
    evaluate.plot_feature_importance(best_model, all_feature_names)

if __name__ == "__main__":
    main()