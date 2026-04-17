# 📘 README — Session 28: Model Implementation Dashboard

## 🎯 Overview

This project contains a Streamlit dashboard that allows you to:

- Load a dataset  
- Select a target variable  
- Train multiple predictive models  
- Compare their performance  
- Visualize results interactively  

The goal is to understand how predictive models can be used as **reusable modules** and compared within a single system.

---

## 📁 Project Structure

```
session28_model_dashboard.py
session28_industrial_model_comparison.csv
requirements.txt
```

---

## ⚙️ Setup Instructions

### 1. Create and activate a virtual environment (recommended)

#### Windows (PowerShell):

```
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux:

```
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## 🚀 How to Run the App

⚠️ IMPORTANT:  
Do **NOT** run the script using `python`.

Instead, use:

```
streamlit run session28_model_dashboard.py
```

---

## 💡 Optional Shortcut

### Option 1 — Create a `.bat` file (Windows)

Create a file named:

```
run_app.bat
```

With this content:

```
@echo off
venv\Scripts\activate
streamlit run session28_model_dashboard.py
pause
```

Then just double-click it.

---

### Option 2 — Create a simple alias (Mac/Linux)

Add this to your terminal config (`.bashrc`, `.zshrc`):

```
alias runapp="streamlit run session28_model_dashboard.py"
```

Then run:

```
runapp
```

---

## 📊 How to Use the Dashboard

1. Upload a CSV file  
2. Select the target variable  
3. Adjust parameters (optional):
   - Test size  
   - Polynomial degree  
   - Tree depth  
   - ANN configuration  
4. Click **"Run Model Comparison"**  

---

## 📈 Output

The dashboard will display:

- Model comparison table (R², MAE, MSE, RMSE)  
- Best performing model  
- R² and RMSE bar charts  
- Predicted vs Actual plot  
- Residual plot  

---

## ⚠️ Common Error

If you run:

```
python session28_model_dashboard.py
```

You may see warnings like:

```
missing ScriptRunContext
```

👉 This is NOT an error — it just means Streamlit is not being used correctly.

✔️ Always use:

```
streamlit run session28_model_dashboard.py
```

---

## 🧠 Key Learning Objective

This dashboard demonstrates that:

> Models are not isolated scripts — they are reusable components that can be integrated, compared, and used for decision-making.
