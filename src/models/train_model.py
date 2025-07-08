# src/models/train_model.py

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter

def train_cox_model(X, y_time, y_event, test_size=0.2, random_state=42,
                    save_dir='outputs/models'):
    """
    Trains a Cox Proportional Hazards model on the given features and survival data.

    Parameters:
    - X: pd.DataFrame of features
    - y_time: pd.Series of survival times
    - y_event: pd.Series of event indicators (1=event occurred, 0=censored)
    - test_size: Fraction of data to use for testing
    - random_state: Seed for reproducibility

    Returns:
    - model: Trained CoxPHFitter model
    - X_test_scaled: Scaled test features
    - y_time_test: Test survival times
    - y_event_test: Test event indicators
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Split the data
    X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
        X, y_time, y_event,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y_event
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Combine for training
    train_df = X_train_scaled.copy()
    train_df['duration'] = y_time_train
    train_df['event'] = y_event_train

    # Train Cox model
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col='duration', event_col='event')

    # Save model and scaler
    joblib.dump(cph, os.path.join(save_dir, "cox_model.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    return cph, scaler, X_test_scaled, y_time_test, y_event_test
