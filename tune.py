import pandas as pd
import numpy as np
import json  # <--- NEW IMPORT
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src import preprocess
from src import feature_engineering

def run_tuning():
    # ... (Same loading and splitting code as before) ...
    print("--- 1. Loading & Cleaning Data for Tuning ---")
    df = preprocess.load_and_clean_data('data/diabetic_data.csv')
    df = feature_engineering.apply_feature_engineering(df)
    
    num_feats, ord_feats, cat_feats, age_order = preprocess.get_feature_lists()
    preprocessor = preprocess.build_pipeline(num_feats, ord_feats, cat_feats, age_order)
    
    X = df.drop(columns=['readmitted_lt_30'])
    y = df['readmitted_lt_30']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ... (Same Pipeline) ...
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', rf)
    ])
    
    # ... (Same Param Dist) ...
    param_dist = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, 30, None],
        'model__min_samples_leaf': [1, 2, 4, 8],
        'model__min_samples_split': [2, 5, 10]
    }
    
    print("--- 2. Starting Randomized Search ---")
    search = RandomizedSearchCV(
        pipe, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=3, 
        scoring='f1', 
        verbose=2, 
        n_jobs=-1, 
        random_state=42
    )
    
    search.fit(X_train, y_train)
    
    print(f"BEST F1 SCORE: {search.best_score_:.4f}")
    
    # --- NEW: CLEAN AND SAVE PARAMETERS ---
    # The pipeline gives params like 'model__n_estimators'. 
    # We need to remove 'model__' so the classifier can understand them.
    best_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
    
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)
        
    print("SUCCESS: Saved best parameters to 'best_params.json'")

if __name__ == "__main__":
    run_tuning()