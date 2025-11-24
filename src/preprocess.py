import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def load_and_clean_data(filepath):
    """
    Loads data, drops high-missingness columns, deduplicates, 
    and creates the binary target variable.
    """
    # Load with '?' as NaN
    print("Loading dataset...")
    df = pd.read_csv(filepath, na_values='?')
    
    # Drop unusable columns
    cols_to_drop = [
        'weight', 'medical_specialty', 'payer_code', 'encounter_id',
        'citoglipton', 'examide'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Deduplicate
    df = df.drop_duplicates(subset=['patient_nbr'], keep='first')
    
    # Target Transformation: 1 if <30, else 0
    df['readmitted_lt_30'] = np.where(df['readmitted'] == '<30', 1, 0)
    df = df.drop(columns=['readmitted'])
    
    # Drop patient_nbr as it's not a feature
    df = df.drop(columns=['patient_nbr'], errors='ignore')
    
    return df

def get_feature_lists():
    """Returns lists of features by type."""
    numeric_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses'
    ]

    age_order = [
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
    ]
    ordinal_features = ['age']

    categorical_features = [
        'race', 'gender', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',
        'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone',
        'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin',
        'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
        'diabetesMed', 'diag_1_group' 
    ]
    
    return numeric_features, ordinal_features, categorical_features, age_order

def build_pipeline(numeric_features, ordinal_features, categorical_features, age_order):
    """Defines the preprocessing steps."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[age_order]))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor