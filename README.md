# 🩺 Survival Curve Prediction for Breast Cancer Patients Using Cox Proportional Hazards Model

This project is a **clinical decision support system** that predicts survival probabilities of breast cancer patients using the **Cox Proportional Hazards model**. By incorporating clinical, demographic, genetic, tumor, and treatment data, it enables oncologists to estimate how different treatment strategies (e.g., chemotherapy, hormone therapy, amputation) impact long-term survival.

## 🚀 Features

- Predict individual patient survival curves over customizable time horizons
- Analyze the impact of clinical interventions on survival probabilities
- Visual output of survival probabilities at critical milestones (3, 5, 10, 15 years)
- Modular code structure for easy integration and model updates
- SHAP-based analysis for model interpretability (in development)
- Future-ready for LLM integration to explain prognosis in natural language

## 🧠 Core Technologies

- `scikit-survival`
- `pandas`, `numpy`
- `matplotlib`
- `scikit-learn`
- `shap` (for exploratory model explainability)

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/ivanglez-6/Survival-Curve-Prediction-for-Breast-Cancer-Patients-Using-Cox-Proportional-Hazards-Model.git
cd Survival-Curve-Prediction-for-Breast-Cancer-Patients-Using-Cox-Proportional-Hazards-Model
```

2. **Set up virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 🧪 Example Usage

You can simulate a patient prediction with the included `test.py` script:

```bash
python test.py
```

This will:
- Define a hypothetical patient with clinical and treatment data
- Apply preprocessing and feature engineering
- Generate the survival curve using a trained Cox model
- Print survival probabilities at 3, 5, 10, and 15 years
- Save the survival curve plot to `outputs/figures/survival_curve_test.png`

### 🧬 Example Patient

```python
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
    'histtype': 1
}])
```

## 📈 Output Example

A typical output survival curve will look like this:

```
Year 3:  95.2% survival probability
Year 5:  89.1%
Year 10: 74.6%
Year 15: 58.3%
```

And the curve image will be saved as:

📁 `outputs/figures/survival_curve_test.png`

*(Make sure the `outputs/figures/` directory exists, but if you cloned the repository, it will)*

## 🧠 SHAP + LLM Integration (WIP)

This project includes groundwork for **SHAP-based explainability** to identify the most influential features per patient. The long-term goal is to **integrate a large language model (LLM)** that provides natural language explanations like:

> “The Cox model predicts a 74.6% survival probability at 10 years. This is primarily influenced by the patient’s tumor size, lymph node involvement, and grade. Hormonal therapy is associated with a positive shift in survival outcome…”

## 📂 Project Structure

```
src/
  ├── data_prep/               # Preprocessing and feature engineering
  ├── predict/                 # Inference pipeline and survival curve generation
  ├── explain/                 # SHAP-based exploratory tools (in development)
outputs/
  ├── models/                  # Trained Cox model and scaler
  ├── predictions/            # CSV with survival probability values
  ├── figures/                # Survival curve visualizations
test.py                        # Example script for inference
```

## 🧑‍⚕️ Intended Audience

This tool is designed for **oncologists and clinical researchers** to aid in understanding and communicating survival probabilities, with the eventual aim of **personalized cancer treatment planning**.

## 📌 Notes

- The model was trained on a curated dataset of breast cancer patients with known treatment and survival outcomes.
- The Cox model supports censoring, and time horizons can be adjusted based on dataset scale.
- Make sure paths to `cox_model.pkl` and `scaler.pkl` are correct when deploying.