# src/models/evaluate_model.py

from lifelines.utils import concordance_index
import pandas as pd

def evaluate_cox_model(model, X_test_scaled, y_time_test, y_event_test):
    """
    Evaluates a trained Cox model using Concordance Index.

    Parameters:
    - model: Trained CoxPHFitter object
    - X_test_scaled: Scaled test features
    - y_time_test: Series of survival times
    - y_event_test: Series of event indicators

    Returns:
    - c_index: Concordance index (float)
    """
    # Prepare test DataFrame
    test_df = X_test_scaled.copy()
    test_df['duration'] = y_time_test
    test_df['event'] = y_event_test

    # Predict partial hazards
    predictions = model.predict_partial_hazard(test_df)

    # Compute C-index
    c_index = concordance_index(
        event_times=y_time_test,
        predicted_scores=-predictions,  # Negative because higher hazard = shorter survival
        event_observed=y_event_test
    )

    return c_index
