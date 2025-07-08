# src/data_prep/preprocess.py

import pandas as pd
import numpy as np

def feature_engineering_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-specific feature engineering:
    - Log transforms 'posnodes'
    - Converts 'histtype' into binary column 'histtype_is_1'
    - Drops identifiers: 'patient', 'ID', and 'barcode'

    Parameters:
    - df: Input DataFrame

    Returns:
    - df_transformed: DataFrame with new features and dropped columns
    """
    df = df.copy()

    # Log transform 'posnodes' safely
    if 'posnodes' in df.columns:
        df['posnodes'] = np.log1p(df['posnodes'])

    # Create binary version of 'histtype' (1 is the positive class)
    if 'histtype' in df.columns:
        df['histtype_is_1'] = (df['histtype'] == 1).astype(int)
        df = df.drop(columns=('histtype'))

    # Drop identifier columns if they exist
    cols_to_drop = ['Patient', 'ID', 'barcode']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Drop the column timerecurrence since its not important for this model
    if 'timerecurrence' in df.columns:
        df = df.drop(columns=('timerecurrence'))   
    
    # Save the feature_engineered_data to a csv file 

    # Split data for future training or inference
    y_time = df['survival'] if 'survival' in df.columns else None
    y_event = df['eventdeath'] if 'eventdeath' in df.columns else None

    X = df.drop(columns=[c for c in ['survival', 'eventdeath'] if c in df.columns])

    return X, y_time, y_event


