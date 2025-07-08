# test.py

import pandas as pd
from src.data_prep.preprocess import feature_engineering_data
from src.predict.inference import generate_survival_curve

def main():
    # === 1. Create a hypothetical patient (raw format) ===
    patient = pd.DataFrame([{
        'age': 55,
        'chemo': 0,
        'hormonal': 1,
        'amputation': 0,
        'diam': 20,
        'posnodes': 4,
        'grade': 3,
        'angioinv': 2,
        'lymphinfil': 1,
        'er_status': 0,
        'histtype': 1,
        'Patient': 'TEST123',
        'ID': 999
    }])

    # === 2. Apply feature engineering ===
    patient_transformed, y, y_t = feature_engineering_data(patient)

    # === 3. Generate and save survival curve ===
    curve = generate_survival_curve(
        patient_df=patient_transformed,
        model_path="outputs/models/cox_model.pkl",
        scaler_path="outputs/models/scaler.pkl",
        time_horizon=19,
        plot=True,
        save_preds_dir='outputs/predictions',
        save_plot_dir='outputs/figures'
    )
    
    for year in [3, 5, 10, 15]:
        prob = curve.iloc[year]['survival_probability']
        print(f"Survival at year {year}: {prob:.4f}")
        
    print("\nSurvival curve for hypothetical patient saved.")

if __name__ == "__main__":
    main()
