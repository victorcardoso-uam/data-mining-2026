# Data Mining Final Project — Chihuahua Climate Analysis
**Universidad Anáhuac Mayab | Periodo 202610**  
**Profesor:** Víctor Cardoso Fernández

---

## Problem Statement
Predict monthly **Earth Skin Temperature (°C)** in the state of Chihuahua, Mexico,
using climate variables: solar irradiance, precipitation, wind speed, and month.

## Dataset
- **Source:** `Data_Chihuahua.csv` (NASA POWER / CONAGUA)
- **Variables:** Surface Irradiance (W/m²), Precipitation (mm), Wind Speed (m/s), Earth Skin Temp (°C)
- **Period:** 2020–2025 | **Coverage:** 3×3 geographic grid over Chihuahua
- **Final rows:** ~142 (monthly observations after reshaping and cleaning)

## Target Variable (y)
`temperature_c` — Monthly Earth Skin Temperature in °C

## Input Variables (X)
| Feature | Description |
|---|---|
| `solar_irradiance_wm2` | Surface solar irradiance (W/m²) |
| `precipitation_mm` | Monthly precipitation (mm) |
| `wind_speed_ms` | Wind speed (m/s) |
| `MONTH_NUM` | Month number (1–12) |

## Project Structure
```
data_mining_project/
├── data/
│   ├── raw/
│   │   └── Data_Chihuahua.csv
│   └── processed/
│       ├── chihuahua_dataset.csv        ← cleaned dataset (output of data_preparation.py)
│       ├── decision_tree_results.csv
│       ├── regression_results.csv
│       ├── ann_results.csv
│       └── final_comparison.csv
├── outputs/
│   └── plots/
│       ├── decision_tree_evaluation.png
│       ├── regression_evaluation.png
│       ├── ann_evaluation.png
│       ├── ann_loss_curves.png
│       ├── final_comparison_metrics.png
│       └── all_models_predicted_vs_actual.png
├── data_preparation.py          ← Step 1: clean and reshape data
├── decision_tree.py             ← Step 2: Decision Tree (default + pruned)
├── regression_models.py         ← Step 3: Linear + Polynomial Regression
├── ann_model.py                 ← Step 4: ANN (3 configurations)
├── final_comparison.py          ← Step 5: Compare all models
├── cluster_analysis.py          ← Session 9 (previous work)
├── market_segmentation.py       ← Session 3.2 (previous work)
├── association_and_sequence_analysis.py  ← Session 3.3 (previous work)
├── requirements.txt
└── README.md
```

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare the dataset (run first)
python data_preparation.py

# 3. Run models (in any order after preparation)
python decision_tree.py
python regression_models.py
python ann_model.py

# 4. Generate final comparison
python final_comparison.py
```

## Models Implemented
| Model | Variations | Key Parameters |
|---|---|---|
| Decision Tree | Default + Pruned | `max_depth=5`, `min_samples_split=10` |
| Linear Regression | Standard | — |
| Polynomial Regression | Degree 2 and Degree 3 | `PolynomialFeatures` |
| ANN (MLPRegressor) | 3 configurations | `hidden_layer_sizes`, `activation`, `max_iter` |

## Evaluation Metrics
All models are evaluated using: **R², MAE, MSE, RMSE** on the same 80/20 train/test split (`random_state=42`).

## Key Findings
- `MONTH_NUM` is the dominant predictor due to Chihuahua's strong seasonal cycle.
- Polynomial Regression (degree=2) captures the seasonal curvature better than linear.
- ANN with deeper configurations captures nonlinear climate interactions most effectively.
- The default Decision Tree overfits; pruning (max_depth=5) improves generalization.
