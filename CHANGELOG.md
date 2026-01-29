# Project Changelog

This document tracks the progress, decisions, and learnings throughout the NHANES Diabetes Prediction project.

---

## [2026-01-28] - Phase 0: Project Setup & Infrastructure

### Objective
Establish the project infrastructure including directory structure, configuration, dependencies, and documentation framework.

### Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Survey Weights** | Descriptive stats only | Use NHANES weights for EDA and population-level statistics, but train unweighted prediction models. This is standard ML practice since we're optimizing for individual-level predictions, not population estimates. |
| **Validation Strategy** | Random stratified split (70/15/15) | Stratified by target class to ensure balanced representation. Random split maximizes training data vs. temporal split. |
| **Data Scope** | Start with 2015-2018 | Begin with recent years for faster iteration (fewer variable naming changes). Expand to 1999-2018 at Phase 7 when we need more data for advanced models. |
| **Experiment Tracking** | MLflow | Local tracking with mlruns/. Logs params, metrics, artifacts, and models. Can migrate to remote server later if needed. |
| **Target Variable** | 3-class + HbA1c regression | Classification: No diabetes / Prediabetes / Diabetes. Regression: Predict HbA1c level. Include medication data (insulin, pills) as diabetes indicators. |

### Implementation

**Directory structure created:**
```
diabetes_prediction_project/
├── config/config.yaml       # Central configuration
├── data/{raw,interim,processed}/
├── notebooks/
├── src/{data,features,models,visualization,utils}/
├── models/{classification,regression}/{with_labs,without_labs}/
├── reports/figures/
├── mlruns/
├── tests/
├── app/
├── requirements.txt
├── .gitignore
└── CHANGELOG.md
```

**Key files created:**
- `requirements.txt`: Pinned versions for reproducibility (pandas, scikit-learn, lightgbm, mlflow, etc.)
- `config/config.yaml`: Central configuration for data paths, thresholds, modeling params
- `.gitignore`: Excludes raw data, mlruns, model artifacts, and sensitive files

### Learnings

1. **Why separate data directories?**
   - `raw/`: Original downloaded data (never modified, gitignored for size)
   - `interim/`: Intermediate processing stages (can regenerate)
   - `processed/`: Final cleaned datasets ready for modeling

2. **Why pin dependency versions?**
   - Ensures reproducibility across environments
   - Prevents breaking changes from updates
   - Makes debugging easier when something fails

3. **Why centralized configuration?**
   - Single source of truth for thresholds, paths, parameters
   - Easy to modify without changing code
   - Documents key project decisions in one place

### Results/Outcomes
- Project infrastructure ready for Phase 1
- All directories and configuration files in place
- Git repository initialized

### Next Steps
- **Phase 1**: Data Acquisition
  - Download NHANES XPT files for 2015-2016 and 2017-2018
  - Create data loading utilities
  - Generate initial data manifest

---
