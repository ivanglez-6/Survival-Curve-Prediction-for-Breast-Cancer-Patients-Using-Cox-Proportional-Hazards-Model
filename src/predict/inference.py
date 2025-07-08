# src/predict/inference.py

import pandas as pd
import joblib
import os
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

def generate_survival_curve(patient_df, model_path, scaler_path, time_horizon=19, plot=False,
                            save_preds_dir='outputs/predictions',
                            save_plot_dir='outputs/figures'):
    """
    Generates a survival curve for a single patient.

    Parameters:
    - patient_df: DataFrame with one row of patient data (raw features)
    - model_path: path to saved Cox model .pkl
    - scaler_path: path to saved scaler .pkl
    - time_horizon: number of time units to simulate (e.g., days/months)
    - plot: whether to plot the survival curve

    Returns:
    - DataFrame with timeline and survival probabilities
    """

    # Load model and scaler
    model: CoxPHFitter = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Scale the patient's data
    X_scaled = pd.DataFrame(
        scaler.transform(patient_df),
        columns=patient_df.columns
    )

    # Predict survival function
    survival_df = model.predict_survival_function(X_scaled, times=range(0, time_horizon))

    # Format result
    survival_curve = pd.DataFrame({
        'timeline': survival_df.index,
        'survival_probability': survival_df.values.flatten()
    })

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(survival_curve['timeline'], survival_curve['survival_probability'], label='Survival Curve')
        plt.title('Predicted Survival Curve')
        plt.xlabel('Years After Diagnosis')
        plt.ylabel('Survival Probability')
        plt.grid(True)
        plt.ylim(0, 1.01)
        plt.savefig(os.path.join(save_plot_dir, "survival_curve_test.png"), dpi=200)
        plt.show()
        

    survival_curve.to_csv(os.path.join(save_preds_dir, "survival_curve_test.csv"))
    
    return survival_curve
