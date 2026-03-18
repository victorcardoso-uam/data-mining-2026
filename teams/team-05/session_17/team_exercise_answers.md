# Session 17 — Team Exercise: Assembly Line Productivity
## Team-05 Answers

**Dataset**: team05_assembly_line_productivity.csv (190 observations)  
**Target**: daily_productivity_units (continuous variable) — Regression problem

---

## Correlations with Target

| Variable | Correlation |
|----------|------------|
| line_speed_units_hr | +0.5831 ⭐ |
| downtime_min | -0.5080 ⭐ |
| workers_count | +0.4903 |
| material_delay_min | -0.2994 |
| inspection_time_min | -0.1367 |

⭐ = Strong correlation (abs > 0.5)

---

## Question 1: Which variables could be used as independent variables?

**Answer:**
All columns except `daily_productivity_units` (the target):

**Numeric predictors:**
- workers_count
- line_speed_units_hr
- downtime_min
- material_delay_min
- inspection_time_min

**Categorical predictors:**
- product_type

**Total: 6 candidate independent variables**

Note: `assembly_line_id` is just an identifier, so exclude it.

---

## Question 2: Which variables should be excluded?

**Answer:**

| Variable | Status | Reason |
|----------|--------|--------|
| assembly_line_id | ❌ EXCLUDE | Identifier only, no predictive value |
| product_type | ✅ KEEP | Categorical predictor, type affects productivity |
| workers_count | ✅ KEEP | More workers → higher productivity |
| line_speed_units_hr | ✅ KEEP | **Strongest predictor** (0.583) |
| downtime_min | ✅ KEEP | **Negative strong correlation** (-0.508) |
| material_delay_min | ✅ KEEP | Weak but relevant |
| inspection_time_min | ✅ KEEP | Weakest correlation but included in process |

**Final recommendation**: Use 6 variables (exclude only `assembly_line_id`)

---

## Question 3: Why do your selected variables make sense for predicting productivity?

**Answer:**

| Variable | Business Logic |
|----------|---|
| **line_speed_units_hr** | Fastest correlation (0.583). Faster lines = more units produced. Clear causal relationship. |
| **downtime_min** | Negative correlation (-0.508). Downtime = lost production. More downtime = less productivity. |
| **workers_count** | More workers typically means higher output (0.490). Direct staffing impact. |
| **product_type** | Different products (Standard/Premium/Customized) likely have different productivity rates. |
| **material_delay_min** | Delays reduce production time. More delays = fewer units (negative). |
| **inspection_time_min** | Quality control takes time. Higher inspection = less throughput. |

**Conclusion**: All selected variables logically impact daily productivity in manufacturing.

---

## Question 4: Would your answer change if the target variable were different?

**Answer: YES**

**Example 1: If target = "product_type"**
- We'd be predicting which product type is produced
- `product_type` would be REMOVED from predictors
- Independent variables would be: workers_count, line_speed_units_hr, downtime_min, material_delay_min, inspection_time_min

**Example 2: If target = "workers_count"**
- Predicting staffing needs
- `workers_count` would be REMOVED
- `daily_productivity_units` would become an independent variable
- Different predictors would be relevant

**Example 3: If target = "downtime_min"**
- Predicting downtime occurrences
- `downtime_min` would be REMOVED
- Would need predictors for downtime causes (machinery age, maintenance schedule, etc.)

**Key insight**: The target determines which variables are relevant. Always exclude the target from independent variables and choose predictors that logically influence your specific goal.

---

**Team-05** | March 12, 2026 | Assembly Line Productivity Analysis
