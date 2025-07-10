from flask import Flask, render_template, request, redirect, url_for
import os
from src.data_prep.preprocess import feature_engineering_data
from src.predict.inference import generate_survival_curve
import shutil
import pandas as pd


app = Flask(__name__)

os.chdir("C:/Users/Usuario/source/repos/Survival-Curve-Prediction-for-Breast-Cancer-Patients-Using-Cox-Proportional-Hazards-Model")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/input", methods=["GET", "POST"])
def input_data():
    if request.method == "POST":
        try:
            patient_data = {
                'age': int(request.form['age']),
                'chemo': int(request.form['chemo']),
                'hormonal': int(request.form['hormonal']),
                'amputation': int(request.form['amputation']),
                'diam': int(request.form['diam']),
                'posnodes': int(request.form['posnodes']),
                'grade': int(request.form['grade']),
                'angioinv': int(request.form['angioinv']),
                'lymphinfil': int(request.form['lymphinfil']),
                'er_status': int(request.form['er_status']),
                'histtype': int(request.form['histtype'])
            }

            patient_data = pd.DataFrame([patient_data])

            # 1. Feature engineering
            patient_transformed, y, y_t = feature_engineering_data(patient_data)

            # 2. Generate survival curve and plot
            _ = generate_survival_curve(
                patient_df=patient_transformed,
                model_path="outputs/models/cox_model.pkl",
                scaler_path="outputs/models/scaler.pkl",
                time_horizon=19,
                plot=True,
                save_preds_dir="outputs/predictions",
                save_plot_dir="outputs/figures"
            )

            # 3. Copy the plot to static folder
            source_path = "outputs/figures/survival_curve_test.png"
            dest_path = "static/survival_curve.png"
            shutil.copyfile(source_path, dest_path)

            # 4. Render result page with image
            return render_template("result.html", patient=patient_data, plot="survival_curve.png")

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("input.html")


if __name__ == "__main__":
    app.run(debug=True)


