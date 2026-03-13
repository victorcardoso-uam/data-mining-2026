# Session 17 — Predictor Selection Analysis
## Team-05: Diabetes Prediction

**Dataset**: diabetes_cleaned.csv (96,146 rows, 8 independent variables)  
**Target**: diabetes (0=No, 1=Yes) — Classification problem

---

## Correlation with Target

1. blood_glucose_level: 0.424
2. HbA1c_level: 0.406
3. age: 0.265
4. bmi: 0.215
5. hypertension: 0.196
6. heart_disease: 0.171
7. gender: (categorical)
8. smoking_history: (categorical)

---

## Questions & Answers

**1. Which variable are you trying to predict?**  
`diabetes` — Binary classification (0=No diabetes, 1=Diabetes)

**2. Which columns are true candidate predictors?**  
All 8 variables: age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, gender, smoking_history

**3. Which columns should be excluded?**  
None. All have medical relevance for diabetes prediction.

**4. Do predictors make sense?**  
Yes. Blood glucose and HbA1c directly measure glucose control (main diabetes indicator). Age, BMI, hypertension, heart_disease are known risk factors. Gender and smoking history are demographic/lifestyle factors.

**5. Are any variables redundant?**  
Potentially: blood_glucose_level and HbA1c_level both measure glucose but capture different timeframes (immediate vs. 3-month average). Use both for better prediction.

**6. Information leakage?**  
No. All variables are measurements available before diagnosis.

**7. Would variables change if target changed?**  
Yes. If predicting "hypertension" instead, different variables would matter. Target determines which predictors are relevant.

---

## Recommendation

**Use all 8 variables** for diabetes prediction:
- Strongest predictors: blood_glucose_level, HbA1c_level
- Supporting predictors: age, bmi, gender, smoking_history
- Additional factors: hypertension, heart_disease

**Rationale**: Medical relevance, no data leakage, moderate correlations, no redundancy.

---

**Team-05** | March 12, 2026
