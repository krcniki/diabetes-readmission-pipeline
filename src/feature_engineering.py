import pandas as pd

def group_diagnosis(diag_code):
    """
    Groups ICD-9 diagnosis codes into broader categories.
    """
    diag_code = str(diag_code)
    if diag_code.startswith('250'):
        return 'Diabetes'
    elif diag_code.startswith('V') or diag_code.startswith('E'):
        return 'External/Injury'
    elif '390' <= diag_code <= '459' or diag_code == '785':
        return 'Circulatory'
    elif '460' <= diag_code <= '519' or diag_code == '786':
        return 'Respiratory'
    elif '520' <= diag_code <= '579' or diag_code == '787':
        return 'Digestive'
    elif '580' <= diag_code <= '629' or diag_code == '788':
        return 'Genitourinary'
    elif '800' <= diag_code <= '999':
        return 'Injury/Poisoning'
    elif '710' <= diag_code <= '739':
        return 'Musculoskeletal'
    else:
        return 'Other'

def apply_feature_engineering(df):
    """
    Applies the grouping logic to the dataframe and handles necessary fillna steps.
    """
    # Fill missing diag_1 with 'Other' before grouping
    df['diag_1'] = df['diag_1'].fillna('Other')
    
    # Apply the grouping function
    print("Engineering 'diag_1_group' feature...")
    df['diag_1_group'] = df['diag_1'].apply(group_diagnosis)
    
    # Drop original diagnosis columns
    df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])
    
    return df