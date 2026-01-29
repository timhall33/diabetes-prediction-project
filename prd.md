# Product Requirements Document (PRD)
# NHANES Diabetes Prediction & Risk Analysis Project

## Project Overview

### Purpose
Build a comprehensive, portfolio-ready machine learning project that predicts diabetes status and risk using NHANES (National Health and Nutrition Examination Survey) data from 1999-2018. This project demonstrates the complete data science lifecycle from raw data acquisition through model deployment, showcasing skills relevant to data science positions.

### Goals
1. **Technical Excellence**: Implement industry best practices across the entire ML pipeline
2. **Educational Value**: Document decisions and learnings throughout the process
3. **Portfolio Impact**: Create a visually compelling, well-documented GitHub repository
4. **Actionable Insights**: Provide interpretable results that inform lifestyle interventions

---

## Phase 0: Project Setup & Infrastructure

### 0.1 Repository Structure
Create the following directory structure:

```
nhanes-diabetes-prediction/
├── README.md                    # Project overview with badges, visuals, results summary
├── prd.md                       # This document
├── CHANGELOG.md                 # Running summary of all work done
├── requirements.txt             # Python dependencies with versions
├── environment.yml              # Conda environment (alternative)
├── .gitignore                   # Standard Python + data science ignores
├── pyproject.toml               # Project metadata
│
├── config/
│   ├── config.yaml              # Central configuration file
│   ├── feature_definitions.yaml # Feature names, mappings, transformations
│   └── model_configs/           # Hyperparameter configurations per model
│
├── data/
│   ├── raw/                     # Original XPT files (gitignored)
│   │   └── {year}/              # Organized by survey year
│   ├── interim/                 # Intermediate processing stages
│   ├── processed/               # Final cleaned datasets
│   └── data_dictionary.md       # Documentation of all variables
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_eda_visualizations.ipynb
│   ├── 06_baseline_models.ipynb
│   ├── 07_model_training_classification.ipynb
│   ├── 08_model_training_regression.ipynb
│   ├── 09_model_evaluation.ipynb
│   ├── 10_feature_importance.ipynb
│   └── 11_final_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py          # NHANES data acquisition
│   │   ├── loaders.py           # XPT file loading utilities
│   │   ├── cleaners.py          # Data cleaning functions
│   │   └── validators.py        # Data validation checks
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── builders.py          # Feature engineering pipeline
│   │   ├── transformers.py      # Custom sklearn transformers
│   │   └── selectors.py         # Feature selection methods
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py             # Training orchestration
│   │   ├── evaluate.py          # Evaluation metrics & comparisons
│   │   ├── tune.py              # Hyperparameter optimization
│   │   └── registry.py          # Model versioning/tracking
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── eda.py               # EDA visualizations
│   │   ├── evaluation.py        # Model performance plots
│   │   └── interpretation.py    # SHAP, feature importance plots
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Logging configuration
│       └── helpers.py           # Miscellaneous utilities
│
├── models/                      # Saved model artifacts
│   ├── classification/
│   │   ├── with_labs/
│   │   └── without_labs/
│   └── regression/
│       ├── with_labs/
│       └── without_labs/
│
├── reports/
│   ├── figures/                 # Publication-ready visualizations
│   ├── model_comparison.md      # Model performance summary
│   └── final_report.md          # Complete project writeup
│
├── app/                         # Deployment artifacts
│   ├── streamlit_app.py         # Interactive web app
│   ├── api/                     # REST API (FastAPI)
│   └── requirements.txt         # App-specific dependencies
│
└── tests/
    ├── test_data.py
    ├── test_features.py
    └── test_models.py
```

### 0.2 Development Environment
- Python 3.10+
- Key packages: pandas, numpy, scikit-learn, lightgbm, tensorflow/pytorch, shap, plotly, seaborn, matplotlib
- Development: jupyter, pytest, black, flake8, pre-commit
- Tracking: mlflow or weights & biases (optional)

### 0.3 CHANGELOG.md Format
Maintain a running log in this format:

```markdown
# Project Changelog

## [Date] - Phase X.X: Brief Title

### Objective
What we set out to accomplish

### Decisions Made
- Decision 1: Rationale
- Decision 2: Rationale

### Implementation
- Step-by-step what was done
- Code/notebooks created
- Challenges encountered and solutions

### Results/Outcomes
- Key findings
- Metrics achieved
- Visualizations generated (with links to figures/)

### Learnings
- What was learned about the data
- What was learned about the techniques
- Industry best practices applied

### Next Steps
- What follows from this work
```

---

## Phase 1: Data Acquisition & Understanding

### 1.1 Data Download Strategy
**Objective**: Programmatically download all required NHANES XPT files from 1999-2018

**Requirements**:
- Create download script that fetches files from CDC NHANES website
- Handle URL pattern changes across survey years
- Implement retry logic and checksum verification
- Store files organized by year: `data/raw/{year}/{filename}.xpt`
- Generate manifest file listing all downloaded files

**Survey Years**: 1999-2000, 2001-2002, 2003-2004, 2005-2006, 2007-2008, 2009-2010, 2011-2012, 2013-2014, 2015-2016, 2017-2018

**Files to Download per Year**:
| Category | File Prefix | Description |
|----------|-------------|-------------|
| Demographics | DEMO | Demographics & sample weights |
| Dietary | DR1TOT | Dietary interview - Day 1 totals |
| Examination | BMX | Body measures |
| Examination | BPX | Blood pressure |
| Laboratory | GHB | Glycohemoglobin |
| Laboratory | GLU | Plasma fasting glucose |
| Laboratory | ALB_CR | Albumin & creatinine - urine |
| Laboratory | TCHOL/HDL | Cholesterol |
| Laboratory | TRIGLY | Triglycerides & LDL |
| Laboratory | CBC | Complete blood count |
| Laboratory | BIOPRO | Biochemistry profile |
| Questionnaire | DIQ | Diabetes questionnaire |
| Questionnaire | ALQ | Alcohol use |
| Questionnaire | BPQ | Blood pressure & cholesterol |
| Questionnaire | CDQ | Cardiovascular health |
| Questionnaire | DBQ | Diet behavior |
| Questionnaire | PAQ | Physical activity |
| Questionnaire | SLQ | Sleep |
| Questionnaire | SMQ | Smoking |
| Questionnaire | WHQ | Weight history |
| Questionnaire | MCQ | Medical conditions |
| Questionnaire | KIQ_U | Kidney conditions |
| Questionnaire | DPQ | Depression screener |
| Questionnaire | RXQ_RX | Prescription medications |

**Note**: File naming conventions vary by year. Create a mapping configuration.

### 1.2 Data Documentation
- Create comprehensive data dictionary documenting every variable
- Map variable name changes across survey years
- Document units, valid ranges, and missing value codes
- Note survey weight requirements for population-level inference

**ASK USER**: Before proceeding with downloads, confirm the specific years and files needed. Ask if they want to start with a subset for faster iteration.

---

## Phase 2: Data Exploration & Quality Assessment

### 2.1 Initial Data Profiling
**Objective**: Understand data structure, quality, and distributions

**Tasks**:
1. Load each XPT file and examine structure
2. Generate summary statistics for all variables
3. Document missing value patterns by variable and year
4. Identify outliers and anomalous values
5. Check variable consistency across years
6. Examine sample sizes per year

**Deliverables**:
- `notebooks/02_data_exploration.ipynb` with detailed profiling
- Missing data heatmaps by variable and year
- Distribution plots for key numeric variables
- Frequency tables for categorical variables

### 2.2 Study Population & Inclusion/Exclusion Criteria

**Inclusion Criteria**:
- Age ≥ 18 years at screening (RIDAGEYR ≥ 18)
- Participated in NHANES survey years 1999-2000 through 2017-2018

**Exclusion Criteria**:
- Age < 18 years (pediatric diabetes has different characteristics)
- Pregnant participants (gestational diabetes is a distinct condition)
  - Use RIDEXPRG (Pregnancy status) = 1 (Yes, positive lab test or self-reported)
  - Note: This variable may only be available for females of childbearing age

**Document in analysis**:
- Create cohort flow diagram showing sample sizes at each exclusion step
- Report demographics of excluded vs included populations

### 2.3 Target Variable Analysis
**Objective**: Define and validate diabetes classification targets

**Classification Target (3-class)**:
| Class | Criteria | Priority |
|-------|----------|----------|
| Diabetes | DIQ010 = 1 (told have diabetes) OR HbA1c ≥ 6.5% OR Fasting glucose ≥ 126 mg/dL | Highest - if ANY diabetes criterion met, classify as diabetes |
| Pre-diabetes | Not diabetes AND (HbA1c 5.7-6.4% OR Fasting glucose 100-125 mg/dL) | Second |
| No diabetes | Does not meet above criteria | Default |

**Important**: Diabetes classification takes precedence. If conflicting indicators exist (e.g., HbA1c in pre-diabetic range but patient reports diabetes diagnosis), classify as diabetic.

**Regression Target**: Glycohemoglobin (HbA1c) level - continuous

**Analysis**:
- Distribution of target classes across years
- Correlation between diagnostic criteria
- Concordance analysis: How often do criteria agree?
- Handle edge cases: conflicting criteria, missing values

---

## Phase 3: Data Cleaning & Preprocessing

### 3.1 Missing Data Strategy
**Objective**: Systematically handle missing values

**Approach**:
1. **Distinguish missing types**:
   - Structurally missing (variable not in survey year)
   - Survey skip patterns (legitimate non-response)
   - True missing (should have data but don't)

2. **Missing analysis**:
   - Calculate missing rates by variable
   - Test MCAR vs MAR vs MNAR assumptions
   - Visualize missing patterns (missingno package)

3. **Imputation strategy**:
   - For features with <5% missing: Simple imputation (median/mode)
   - For features with 5-30% missing: Multiple imputation (MICE)
   - For features with >30% missing: Consider dropping or special indicators
   - Create missing indicator variables for potentially informative missingness

**ASK USER**: Review missing data patterns and confirm imputation approach before implementation.

### 3.2 Variable Harmonization
**Objective**: Standardize variables across survey years

**Tasks**:
1. Map variable name changes (e.g., ALQ120Q → ALQ121)
2. Harmonize coding schemes (convert to consistent categories)
3. Standardize age capping (≥80 for all years per your specification)
4. Handle unit changes if any
5. Create crosswalk documentation

**Specific Harmonizations Needed** (from your feature list):
- ALQ (Alcohol): Combine ALQ120Q/ALQ120U → ALQ121 format
- DBD (Diet): Harmonize DBD091/DBD090 → DBD895
- PAQ (Physical Activity): Multiple variable mappings needed
- SLQ (Sleep): Two different questionnaire versions
- SMQ (Smoking): SMD080/SMD641 and SMD090/SMD650 mappings
- MCQ (Medical conditions): MCQ300C/MCQ250A, MCQ160L/MCQ500
- KIQ (Kidney): KIQ020/KIQ022

### 3.3 Data Quality Checks
Implement validation rules:
- Range checks (e.g., BMI 10-100, age 0-80)
- Logical consistency (e.g., weight at 25 ≤ heaviest weight)
- Cross-variable validation (e.g., if taking diabetes medication, should have diabetes indicator)

---

## Phase 4: Feature Engineering

### 4.1 Derived Features
Create the following engineered features:

**Calculated Features**:
| Feature | Formula | Rationale |
|---------|---------|-----------|
| TOTAL_WATER | DR1_320Z + DR1_330Z + DR1BWATZ | Total water intake |
| AVG_SYS_BP | mean(BPXSY1, BPXSY2, BPXSY3) | Average systolic BP |
| AVG_DIA_BP | mean(BPXDI1, BPXDI2, BPXDI3) | Average diastolic BP |
| ACR_RATIO | URXUMA / URXUCR | Albumin-creatinine ratio |
| WEIGHT_CHANGE_10YR | BMXWT - (WHD110 * 0.453592) | Weight change from 10yrs ago |
| WEIGHT_CHANGE_25 | BMXWT - (WHD120 * 0.453592) | Weight change from age 25 |
| WEIGHT_FROM_MAX | (WHD140 * 0.453592) - BMXWT | Difference from heaviest |
| WAKE_TIME_DIFF | SLQ330 - SLQ310 | Weekend vs weekday wake time |

**Interaction Features** (consider):
- BMI × Age
- Physical activity × Sedentary time
- Caloric intake × BMI
- Family history × Age

**Ratio Features**:
- Waist-to-height ratio
- Fat-to-protein intake ratio
- Saturated fat percentage of total fat

### 4.2 Feature Categories
Organize features into groups for analysis:

**Demographic** (2):
- Age (RIDAGEYR), Gender (RIAGENDR)

**Anthropometric** (4):
- BMI, Weight, Height, Waist circumference

**Blood Pressure** (2):
- Average systolic, Average diastolic

**Dietary** (16):
- Energy, Protein, Carbohydrates, Sugars, Fiber
- Total/Saturated/Mono/Poly fats
- Sodium, Caffeine, Alcohol, Water intake

**Laboratory** (14):
- Lipid panel: Total cholesterol, HDL, LDL, Triglycerides
- Kidney function: Urine albumin, creatinine, ACR ratio, Serum creatinine
- Liver function: ALT, AST, GGT
- Blood count: WBC, Hematocrit, Hemoglobin, Platelets

**Lifestyle** (variable):
- Alcohol consumption, Smoking history
- Physical activity metrics
- Sleep duration/patterns
- Diet quality indicators

**Medical History** (10+):
- Hypertension, High cholesterol
- Cardiovascular conditions
- Family history of diabetes
- Kidney conditions, Liver conditions

**Mental Health** (5):
- Depression screener items

### 4.3 Feature Preparation for Modeling

**Numerical Features**:
- Standardization (StandardScaler) for linear models
- Keep raw for tree-based models
- Handle outliers (winsorization at 1st/99th percentile)

**Categorical Features**:
- One-hot encoding for nominal variables
- Ordinal encoding for ordered categories
- Target encoding (with caution for leakage)

**Create Two Feature Sets**:
1. **With Labs**: All features including laboratory data
2. **Without Labs**: Exclude lab values (simulates real-world screening without blood tests)

---

## Phase 5: Exploratory Data Analysis (EDA)

### 5.1 Univariate Analysis
**Visualizations to create**:
- Histograms/KDE plots for all numeric features
- Bar charts for categorical features
- Target variable distribution

### 5.2 Bivariate Analysis
**Visualizations**:
- Feature distributions by diabetes status (violin plots, box plots)
- Correlation heatmap (overall and by feature category)
- Scatter plots for highly correlated features
- Chi-square analysis for categorical vs target

### 5.3 Multivariate Analysis
**Visualizations**:
- Pair plots for key feature groups
- PCA visualization (2D/3D)
- t-SNE/UMAP for clustering patterns
- Feature clustering dendogram

### 5.4 Publication-Quality Figures
Create these specific visualizations for reports/README:

1. **Cohort Flow Diagram**: Sample selection and exclusion criteria
2. **Missing Data Pattern**: Heatmap across years and variables
3. **Target Distribution**: Pie/bar chart with sample sizes
4. **Feature Importance Preview**: Top features by correlation with target
5. **Demographic Summary**: Population characteristics table/visualization
6. **Key Risk Factors**: Forest plot or similar showing univariate associations

**ASK USER**: Review EDA findings before proceeding. Discuss any surprising patterns or data quality concerns.

---

## Phase 6: Model Development - Baseline

### 6.1 Data Splitting Strategy
**Approach**:
- Train/Validation/Test split: 70/15/15
- Stratified by target variable
- Consider temporal validation (train on earlier years, test on later)
- Set random seed for reproducibility

### 6.2 Baseline Models
Establish performance benchmarks with simple models:

**Classification**:
- Majority class predictor (naive baseline)
- Logistic regression with default parameters
- Decision tree with limited depth

**Regression**:
- Mean predictor (naive baseline)
- Linear regression
- Simple decision tree

**Metrics to Track**:
| Task | Primary | Secondary |
|------|---------|-----------|
| Classification | Macro F1-Score | Accuracy, Per-class Precision/Recall, AUC-ROC (one-vs-rest) |
| Regression | RMSE | MAE, R², MAPE |

---

## Phase 7: Model Development - Advanced

### 7.1 Classification Models

**Model 1: Logistic Regression**
- Regularization: L1, L2, ElasticNet
- Tune: C (regularization strength), penalty type
- Multi-class: one-vs-rest or multinomial

**Model 2: LightGBM Classifier**
- Native multi-class support
- Key hyperparameters: learning_rate, num_leaves, max_depth, n_estimators, reg_alpha, reg_lambda, min_child_samples
- Handle class imbalance: class_weight or scale_pos_weight

**Model 3: MLP Classifier (sklearn)**
- Architecture: 2-3 hidden layers
- Hyperparameters: hidden_layer_sizes, activation, alpha (L2), learning_rate

**Model 4: Deep Learning (TensorFlow/PyTorch)**
- Architecture: 3-5 layers with dropout
- Batch normalization
- Early stopping
- Class weights for imbalance

### 7.2 Regression Models

**Model 1: Linear Regression**
- Ridge, Lasso, ElasticNet variants
- Tune: alpha (regularization strength)

**Model 2: LightGBM Regressor**
- Same hyperparameter space as classifier
- Objective: regression (MSE) or regression_l1 (MAE)

**Model 3: MLP Regressor (sklearn)**
- Similar to classifier architecture
- Linear output activation

**Model 4: Deep Learning (TensorFlow/PyTorch)**
- Similar architecture to classifier
- Single output neuron, no activation

### 7.3 Hyperparameter Tuning Strategy

**Given compute constraints (M3 MacBook Pro)**:
- Use Optuna or scikit-optimize (Bayesian optimization)
- Limit trials: 50-100 per model
- Use 3-fold CV (not 5) for faster iteration
- Start with coarse search, then refine

**Time Budget Estimates**:
| Model | Estimated Time per Trial | Max Trials | Total Time |
|-------|--------------------------|------------|------------|
| Logistic Regression | 5-10 sec | 50 | ~10 min |
| LightGBM | 30-60 sec | 100 | ~1 hour |
| MLP (sklearn) | 1-2 min | 50 | ~1.5 hours |
| Deep Learning | 5-10 min | 30 | ~5 hours |

**ASK USER**: Confirm acceptable training times. Offer to reduce search space if needed.

### 7.4 Experiment Tracking
Track for each experiment:
- Model type and hyperparameters
- Feature set (with/without labs)
- Training and validation metrics
- Training time
- Random seed

---

## Phase 8: Model Evaluation & Comparison

### 8.1 Classification Evaluation

**Per-Model Metrics**:
- Confusion matrix (3x3)
- Classification report (precision, recall, F1 per class)
- ROC curves (one-vs-rest)
- Precision-Recall curves
- Calibration curves (reliability diagrams)

**Cross-Model Comparison**:
- Performance summary table
- ROC curve overlay plot
- Statistical significance testing (McNemar's test or similar)
- Error analysis: What does each model get wrong?

### 8.2 Regression Evaluation

**Per-Model Metrics**:
- Residual plots
- Predicted vs Actual scatter
- Error distribution histogram
- RMSE, MAE, R², MAPE

**Cross-Model Comparison**:
- Metrics comparison table
- Residual comparison plots
- Bland-Altman plots for model agreement

### 8.3 With-Labs vs Without-Labs Comparison
Critical analysis comparing:
- Performance drop when removing labs
- Which non-lab features compensate
- Clinical implications

### 8.4 Publication-Quality Evaluation Figures

1. **Model Performance Comparison**: Bar chart of key metrics across models
2. **Confusion Matrix Heatmaps**: Side-by-side for top models
3. **ROC Curve Comparison**: All models on one plot
4. **Calibration Plot**: Actual vs predicted probabilities
5. **Performance by Subgroup**: Stratified by age/gender
6. **With/Without Labs Comparison**: Clear visual of performance gap

---

## Phase 9: Model Interpretation & Insights

### 9.1 Feature Importance Analysis

**Methods**:
- Permutation importance (model-agnostic)
- SHAP values (TreeExplainer for LightGBM, DeepExplainer for neural nets)
- Coefficient analysis for linear models

**Visualizations**:
- Global feature importance bar charts
- SHAP summary plots
- SHAP dependence plots for top features
- Partial dependence plots

### 9.2 Actionable Insights

**Risk Factor Analysis**:
- Identify top modifiable risk factors
- Quantify impact of each factor
- Compare lab-based vs lifestyle-based predictors

**Lifestyle Intervention Insights**:
- What dietary changes most impact risk?
- How much does physical activity matter?
- Sleep and diabetes connection
- Weight management importance

**Visualizations**:
- "What-If" analysis plots
- Risk reduction scenarios
- Feature value → risk level charts

### 9.3 Publication-Quality Interpretation Figures

1. **Top 20 Features**: Importance plot with confidence intervals
2. **SHAP Summary**: Beeswarm plot for all features
3. **Key Feature Deep Dives**: SHAP dependence for top 5 features
4. **Risk Factor Infographic**: Visual summary for non-technical audience
5. **Modifiable vs Non-Modifiable**: Grouped importance comparison

---

## Phase 10: Deployment

### 10.1 Streamlit Application

**Features**:
- Input form for user health metrics
- Risk prediction with confidence
- Personalized recommendations
- Feature importance explanation for their prediction
- SHAP force plot for individual predictions

**Pages**:
1. **Risk Calculator**: Input metrics → get prediction
2. **Model Information**: How it works, data sources, limitations
3. **Population Insights**: Key findings from analysis
4. **Compare Scenarios**: What-if analysis tool

### 10.2 API Endpoint (Optional)

**FastAPI Implementation**:
- `/predict/classification`: Returns diabetes risk category + probability
- `/predict/regression`: Returns predicted HbA1c
- `/explain`: Returns feature contributions for prediction

### 10.3 Model Packaging
- Save final models with joblib/pickle
- Include preprocessing pipeline
- Version models clearly
- Document input requirements

---

## Phase 11: Documentation & Polish

### 11.1 README.md Structure

```markdown
# 🩺 Diabetes Risk Prediction Using NHANES Data

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
[Additional badges: stars, forks, last commit]

## 📋 Overview
[2-3 sentence project summary]
[Hero visualization - key finding or model performance]

## 🎯 Key Results
[Performance metrics table]
[Most interesting finding]

## 📊 Dataset
[NHANES description, years used, sample size]

## 🔬 Methodology
[High-level pipeline diagram]
[Brief description of approach]

## 🚀 Quick Start
[Installation and basic usage]

## 📁 Repository Structure
[Folder structure overview]

## 📈 Results & Visualizations
[Gallery of key figures with descriptions]

## 🔮 Try the Model
[Link to Streamlit app or instructions]

## 📚 Detailed Documentation
[Links to notebooks and reports]

## 🤝 Contributing
[Contribution guidelines]

## 📄 License
[License information]

## 📧 Contact
[Your information]
```

### 11.2 Final Report
Create `reports/final_report.md` with:
- Executive summary
- Detailed methodology
- Complete results
- Discussion and limitations
- Future work
- References

---

## Interaction Guidelines for Claude Code

### When to Ask User Questions (AskUserQuestionTool)

**DO ask before**:
- Downloading large amounts of data
- Making irreversible decisions (e.g., dropping features)
- Choosing between multiple valid approaches
- Long-running operations (training, hyperparameter tuning)
- Confirming target variable definitions
- When code produces unexpected results

**DON'T ask for**:
- Standard implementation details
- Code formatting preferences
- Obvious next steps in the workflow

### CHANGELOG Updates
Update CHANGELOG.md after completing each phase or significant step. Include:
- Date and phase
- What was done
- Decisions made with rationale
- Results/outputs
- Learnings
- Next steps

### Code Quality Standards
- Use type hints in Python functions
- Include docstrings for all functions
- Follow PEP 8 style guidelines
- Create unit tests for data transformations
- Log important operations

### Compute-Aware Development
- Profile code before running expensive operations
- Start with data samples for development
- Save intermediate results to avoid recomputation
- Use appropriate batch sizes for neural networks
- Monitor memory usage with large datasets

---

## Success Criteria

### Technical
- [ ] All models trained and evaluated
- [ ] With/without labs comparison complete
- [ ] Performance exceeds naive baseline significantly
- [ ] No data leakage in pipeline
- [ ] Reproducible results (random seeds set)

### Documentation
- [ ] CHANGELOG captures full journey
- [ ] README is compelling and complete
- [ ] Code is well-documented
- [ ] All visualizations are publication-quality

### Portfolio Impact
- [ ] Clear problem statement and business value
- [ ] Demonstrates end-to-end ML skills
- [ ] Shows software engineering best practices
- [ ] Includes deployment component
- [ ] Provides actionable insights, not just predictions

---

## Appendix: Feature Reference

[Include the full feature list from your document here, formatted as a reference]

### Variable Harmonization Quick Reference

| Modern Variable | Legacy Variable(s) | Notes |
|-----------------|-------------------|-------|
| ALQ121 | ALQ120Q + ALQ120U | Alcohol frequency |
| DBD895 | DBD091, DBD090 | Meals not home prepared |
| PAQ605 | PAD200 | Vigorous work activity |
| PAQ620 | PAD320 | Moderate work activity |
| PAQ640 | PAD020 + PAQ050Q | Walking/bicycling |
| SLD012/SLD013 | SLD010H | Sleep hours |
| SMD641 | SMD080 | Smoking days past 30 |
| SMD650 | SMD090 | Cigarettes per day |
| MCQ250A | MCQ300C | Family diabetes history |
| MCQ500 | MCQ160L | Liver condition |
| KIQ022 | KIQ020 | Kidney condition |
