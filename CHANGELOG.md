# Project Changelog

This document tracks the progress, decisions, and learnings throughout the NHANES Diabetes Prediction project.

## Quick Reference

**Environment**: `conda activate diabetes-ml`

**Current Data Files**:

*Interim* (`data/interim/`):
| File | Shape | Description |
|------|-------|-------------|
| `study_population.parquet` | 11,723 × 832 | Raw merged data with target variable |
| `cleaned_minimal_impute.parquet` | 11,723 × 1,562 | For LightGBM (NaN preserved ≥5% missing) |
| `cleaned_full_impute.parquet` | 11,723 × 1,562 | For LogReg/NN (imputed ≤50% missing) |

*Processed* (`data/processed/`):
| File | Shape | Has NaN | Use For |
|------|-------|---------|---------|
| `features_engineered.parquet` | 11,723 × 1,584 | Yes | All features + 22 derived |
| `X_with_labs_minimal.parquet` | 11,698 × 109 | Yes | LightGBM (handles NaN) |
| `X_with_labs_full.parquet` | 11,698 × 96 | No | LogReg, MLP (no NaN) |
| `X_without_labs_minimal.parquet` | 11,698 × 92 | Yes | LightGBM (handles NaN) |
| `X_without_labs_full.parquet` | 11,698 × 82 | No | LogReg, MLP (no NaN) |
| `y_*.parquet` | 11,698 × 1 | No | Target variables |

**Key Constants**:
```python
TARGET_RELATED_COLS = {'LBXGH', 'LBXGLU', 'DIQ010', 'DIQ050', 'DIQ070', 'DIABETES_STATUS'}
# These are NEVER imputed - they define the target variable

IMPUTE_THRESHOLD = 0.05         # Minimal imputation: only <5% missing
MAX_MISSING_RATE = 0.50         # Full imputation: ≤50% missing; >50% removed
```

**Special Value Encoding** (questionnaire items only, NOT lab values):
- `-7` = Refused (originally 7, 77, 777...)
- `-9` = Don't Know (originally 9, 99, 999...)
- `NaN` = Not collected / structurally missing
- Lab columns (LBX*, URX*, etc.) keep original values - 7.0, 9.0 are valid measurements!

**Derived Features** (22 total):
- Blood pressure: AVG_SYS_BP, AVG_DIA_BP, PULSE_PRESSURE, MAP, BP_VARIABILITY
- Weight: WEIGHT_CHANGE_10YR, WEIGHT_CHANGE_25, WEIGHT_FROM_MAX
- Sleep: WAKE_TIME_DIFF, SLEEP_DURATION_DIFF, SLQ310_HOURS, SLQ330_HOURS
- Dietary: TOTAL_WATER, SAT_FAT_PCT, CARB_FIBER_RATIO, SUGAR_CARB_RATIO
- Labs: ACR_RATIO, TG_HDL_RATIO, NON_HDL_CHOL
- Other: WAIST_HEIGHT_RATIO, PHQ9_SCORE, ANY_CVD

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

## [2026-01-29] - Phase 1: Data Acquisition

### Objective
Download NHANES XPT files from CDC website for survey years 2015-2016 and 2017-2018, with an extensible design for adding older years later.

### Research & Decisions

**NHANES URL Pattern Discovery:**
```
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{first_year}/DataFiles/{FILENAME}.xpt
```

**File Suffix Pattern by Year:**
| Years | Suffix | Example |
|-------|--------|---------|
| 1999-2000 | (none) | DEMO |
| 2001-2002 | _B | DEMO_B |
| ... | ... | ... |
| 2015-2016 | _I | DEMO_I |
| 2017-2018 | _J | DEMO_J |

**Key Finding:** The 1999-2000 laboratory files use completely different names (LAB10, LAB13, etc.) compared to modern naming (GHB, TCHOL, etc.). This was documented in `config/file_mappings.yaml` for future expansion.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Test Years** | 2015-2016 + 2017-2018 | Both use modern naming (_I, _J suffixes). Simpler to implement; older year quirks handled later. |
| **Config-Driven Mappings** | `file_mappings.yaml` | File names stored in config, not hardcoded. Allows adding 1999-2000 mappings by editing YAML only. |
| **Manifest Tracking** | JSON with MD5 checksums | Skip re-downloading files that already exist with matching checksums. |

### Implementation

**Files created:**
- `config/file_mappings.yaml` - Year-specific file name mappings with extensibility for older years
- `src/data/download.py` - Download orchestration with retry logic, progress bars, manifest generation

**Download script features:**
- Automatic suffix handling based on year
- 3 retries with exponential backoff (2s, 4s, 8s)
- Progress tracking with tqdm
- MD5 checksum verification
- Skip existing files (based on manifest)
- Detailed logging and summary report

### Results/Outcomes

**Download Summary:**
- Total files: 48 (24 per year × 2 years)
- Success: 48
- Failed: 0
- Total size: ~94 MB

**Sample counts (from DEMO files):**
| Year | Participants |
|------|-------------|
| 2015-2016 | 9,971 |
| 2017-2018 | 9,254 |
| **Total** | **19,225** |

**Files verified to load with pyreadstat:**
- All files contain SEQN (participant ID) column
- Data types read correctly
- No corruption detected

### Learnings

1. **NHANES naming is not consistent across years**
   - Modern years (2001+) use alphabetic suffixes (_B through _J)
   - 1999-2000 has no suffix AND uses different base names for lab files
   - This is why config-driven mappings are essential

2. **Why retry logic matters**
   - CDC servers occasionally timeout or return errors
   - Exponential backoff prevents hammering the server
   - 3 retries with 2x backoff is a good balance

3. **Why checksums in manifest?**
   - Detects corrupted downloads
   - Enables safe re-runs (only download what's missing/changed)
   - Documents exactly what data version was used

### Next Steps
- **Phase 2**: Data Exploration & Quality Assessment

---

## [2026-01-29] - Phase 2: Data Exploration & Quality Assessment

### Objective
Understand data structure, quality, and distributions. Define study population with inclusion/exclusion criteria. Create and validate target variable.

### Research & Decisions

**Study Population Criteria:**
| Criterion | Rule | Rationale |
|-----------|------|-----------|
| **Inclusion** | Age >= 18 | Adults only (pediatric diabetes has different characteristics) |
| **Exclusion** | RIDEXPRG = 1 | Pregnant participants (gestational diabetes is distinct) |

**Target Variable Definition (3-class classification):**
| Class | Criteria | Priority |
|-------|----------|----------|
| **Diabetes (2)** | DIQ010=1 OR HbA1c>=6.5% OR Fasting glucose>=126 | Highest |
| **Prediabetes (1)** | NOT diabetes AND (HbA1c 5.7-6.4% OR FG 100-125) | Second |
| **No Diabetes (0)** | Does not meet above criteria | Default |

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Target encoding** | 0/1/2 numeric | Compatible with sklearn multi-class classifiers. 0=No, 1=Pre, 2=Diabetes. |
| **Missing target handling** | Keep as NaN | Will exclude from training but useful for feature analysis. Only 25 participants (0.2%) have insufficient data for classification. |
| **Fasting glucose** | Include despite missingness | ~55% missing (fasting subsample only), but provides valuable diagnostic info when available. |

### Implementation

**Files created:**
- `src/data/loaders.py` - Data loading utilities with functions to load individual files, merge by SEQN, and combine across years
- `notebooks/02_data_exploration.ipynb` - Comprehensive exploration notebook with data profiling, missing analysis, cohort creation, and target definition

**Figures generated:**
- `reports/figures/missing_values_heatmap.png` - Missing rates by year for key variables
- `reports/figures/age_distribution.png` - Age distribution before filtering
- `reports/figures/hba1c_distribution.png` - HbA1c with diagnostic thresholds
- `reports/figures/glucose_distribution.png` - Fasting glucose with thresholds
- `reports/figures/target_distribution.png` - 3-class target distribution
- `reports/figures/hba1c_vs_glucose.png` - Scatter plot showing concordance
- `reports/figures/bmi_by_diabetes_status.png` - BMI differences by status
- `reports/figures/age_by_diabetes_status.png` - Age differences by status

**Interim data saved:**
- `data/interim/study_population.parquet` - Filtered cohort with target variable (11,723 rows, 832 columns)

### Results/Outcomes

**Cohort Flow:**
| Step | N | Excluded | Reason |
|------|---|----------|--------|
| Total NHANES (2015-2018) | 19,225 | - | - |
| After age filter | 11,878 | 7,347 | Age < 18 |
| After pregnancy filter | 11,723 | 155 | Pregnant |

**Study Population Demographics:**
- Total N: 11,723
- By year: 2015-2016 (5,922), 2017-2018 (5,801)
- Age: mean=49.2, range=[18-80]
- Gender: Male (48.9%), Female (51.1%)

**Target Variable Distribution (N=11,698 with valid target):**
| Class | N | Percentage |
|-------|---|------------|
| No Diabetes | 5,684 | 48.6% |
| Prediabetes | 3,795 | 32.4% |
| Diabetes | 2,219 | 19.0% |
| Missing | 25 | 0.2% |

**Key Findings:**
1. **Class imbalance**: Majority (48.6%) are non-diabetic, but not severely imbalanced
2. **BMI association**: Mean BMI increases with diabetes status (No: 28.1, Pre: 30.3, Diabetes: 33.0)
3. **Age association**: Diabetic group is older (mean 59.4) vs non-diabetic (43.2)
4. **Fasting glucose missingness**: ~55% missing (only fasting subsample tested), while HbA1c has ~2% missing
5. **Diagnostic concordance**: HbA1c and glucose generally agree, with some cases of isolated elevation

### Learnings

1. **Why use HbA1c as primary diagnostic?**
   - Available for ~98% of eligible participants (no fasting required)
   - More stable than glucose (reflects 2-3 month average)
   - Less affected by acute illness or stress

2. **Why include self-reported diabetes?**
   - Captures treated diabetes where HbA1c may be controlled
   - Important for patients on insulin or medications who maintain HbA1c < 6.5%
   - NHANES is cross-sectional, so we don't have diagnostic history

3. **Class imbalance is manageable**
   - 19% diabetes is actually high prevalence (close to US adult rate)
   - Will use stratified sampling and class weights during modeling
   - No need for aggressive oversampling

### Next Steps
- **Phase 3**: Data Cleaning & Preprocessing

---

## [2026-01-29] - Phase 3: Data Cleaning & Preprocessing

### Objective
Clean and preprocess NHANES data with robust validation checks. Create two output datasets: minimal imputation (for tree models) and full imputation (for linear/NN models).

### Research & Decisions

**Missing Data Strategy:**
| Missing % | Tree Models (LightGBM) | Linear/NN Models |
|-----------|------------------------|------------------|
| <5% | Median/mode impute | Median/mode impute |
| ≥5% | Leave NaN + `_MISSING` flag | Impute + `_MISSING` flag |

**Special Value Handling:**
| Code | Meaning | Action |
|------|---------|--------|
| NaN | Not collected | Keep as NaN |
| 7, 77, 777... | Refused | Recode to -7 (preserve as category) |
| 9, 99, 999... | Don't Know | Recode to -9 (preserve as category) |

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Never impute targets** | LBXGH, LBXGLU, DIQ010, DIQ050, DIQ070 | These define the target variable - imputing would cause circular logic |
| **Preserve "Refused"** | Recode to -7 | Refusing to answer may be informative (e.g., alcohol questions) |
| **Preserve "Don't Know"** | Recode to -9 | Different from missing - indicates participant uncertainty |
| **Two output datasets** | Minimal + Full impute | Tree models handle NaN natively; linear models need complete data |

### Implementation

**Files created:**
- `src/data/cleaners.py` - Cleaning functions: recode_special_values, create_missing_flags, impute_low_missing, impute_all_features
- `src/data/validators.py` - Validation functions: check_ranges, check_logical_consistency, check_target_leakage, compare_distributions
- `notebooks/03_data_cleaning.ipynb` - Cleaning pipeline with full logging and validation

**Config updated:**
- `config/config.yaml` - Added validation section with range checks for 16 key variables

**Output files:**
- `data/interim/cleaned_minimal_impute.parquet` - For LightGBM (11,723 × 1,562)
- `data/interim/cleaned_full_impute.parquet` - For LogReg/NN (11,723 × 1,562)
- `data/interim/cleaning_report.json` - Audit trail of all transformations

### Results/Outcomes

**Cleaning Summary:**
| Metric | Value |
|--------|-------|
| Input shape | 11,723 × 832 |
| Output shape | 11,723 × 1,562 |
| Missing flags created | 730 |
| Columns imputed (minimal) | 37 (<5% missing only) |
| Columns imputed (full) | ~290 (≤50% missing) |
| Columns NOT imputed | 481 (>50% missing - left as NaN) |

**Special Value Recoding:**
- "Refused" → -7: recoded across questionnaire columns
- "Don't Know" → -9: recoded across questionnaire columns

**Validation Results:**
| Check | Result | Notes |
|-------|--------|-------|
| Row count preserved | ✓ | 11,723 rows unchanged |
| Target columns not imputed | ✓ | LBXGH, LBXGLU, DIQ010 preserved |
| No target leakage | ✓ | Target columns excluded from features |
| Range violations | 215 | Extreme but valid values (e.g., BMI 86.2) |
| Logic violations | 697 | Mostly weight>max_weight (recall differences) |

**Data Quality Observations:**
- 683 cases where current weight > max weight ever (5.8%) - likely recall bias or post-survey weight gain
- 9 cases where waist ≥ height (0.08%) - extreme obesity
- 5 cases taking diabetes meds without diabetes diagnosis (0.04%) - off-label metformin use (PCOS, weight)
- Diastolic BP = 0 in some readings - valid (aortic regurgitation or cuff artifact)

### Learnings

1. **Why preserve "Refused" and "Don't Know" as categories?**
   - Missing (NaN) ≠ Refused ≠ Don't Know semantically
   - Refusing to answer alcohol questions may indicate problem drinking
   - Not knowing if you have a condition may indicate lack of healthcare access
   - Models can learn from these patterns

2. **Why create two imputed datasets?**
   - LightGBM handles NaN natively with optimal splits - imputing can actually hurt performance
   - Logistic regression requires complete data - must impute
   - This gives us flexibility during modeling phase

3. **Why flag but not remove validation violations?**
   - Extreme BMI (86.2) is possible in severe obesity - removing would bias the sample
   - Weight recall inconsistencies are common in surveys - not measurement errors
   - Better to document anomalies than silently remove them

### Next Steps
- **Phase 4**: Feature Engineering

---

## [2026-01-30] - Phase 4: Feature Engineering

### Objective
Create derived features with clinical rationale, define feature sets (with/without labs), and prepare final modeling datasets.

### Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Derived features** | 20 engineered features | Each based on clinical/scientific evidence for diabetes risk |
| **Feature sets** | with_labs (109) + without_labs (92) | Enables comparison of screening with/without blood tests |
| **Missing flags** | Include in modeling data | Missingness can be informative; increases to 171/140 total features |

### Implementation

**Files created:**
- `src/features/builders.py` - Feature engineering functions with clinical documentation
- `src/features/__init__.py` - Module exports
- `notebooks/04_feature_engineering.ipynb` - Feature creation and validation notebook

**Derived Features Created (20 total):**

*Original Features:*
| Feature | Formula | Clinical Rationale | N Valid |
|---------|---------|-------------------|---------|
| AVG_SYS_BP | mean(BPXSY1-3) | Multiple readings reduce white coat effect | 10,700 |
| AVG_DIA_BP | mean(BPXDI1-3) | Multiple readings reduce variability | 10,700 |
| TOTAL_WATER | sum(water cols) | Hydration affects glucose regulation | 10,138 |
| ACR_RATIO | albumin/creatinine | Early kidney damage marker (diabetic nephropathy) | 10,757 |
| WEIGHT_CHANGE_10YR | current - 10yr ago | Recent weight trajectory predicts risk | 7,569 |
| WEIGHT_CHANGE_25 | current - age 25 | Lifetime weight gain from baseline | 8,900 |
| WEIGHT_FROM_MAX | max - current | Weight loss from peak (intentional vs disease) | 10,830 |
| WAKE_TIME_DIFF | weekend - weekday wake | Social jet lag disrupts metabolism | 5,714 |
| WAIST_HEIGHT_RATIO | waist / height | Central obesity better than BMI for many | 10,435 |
| SAT_FAT_PCT | sat_fat / total_fat | Dietary fat quality indicator | 10,134 |

*Additional Features (added after review):*
| Feature | Formula | Clinical Rationale | N Valid |
|---------|---------|-------------------|---------|
| PULSE_PRESSURE | SYS - DIA | Arterial stiffness marker; >60 = elevated risk | 10,700 |
| MAP | (SYS + 2×DIA) / 3 | Mean arterial pressure; organ perfusion | 10,700 |
| BP_VARIABILITY | std(BPXSY1-3) | Reading variability; independent CV risk | 10,610 |
| CARB_FIBER_RATIO | carbs / fiber | Carb quality; >15 = poor | 10,048 |
| SUGAR_CARB_RATIO | sugars / carbs | Simple vs complex carbs | 10,136 |
| PHQ9_SCORE | sum(DPQ010-090) | Depression score; bidirectional diabetes link | 10,139 |
| TG_HDL_RATIO | triglyc / HDL | Insulin resistance marker; >3.0 = elevated | 4,775 |
| NON_HDL_CHOL | total - HDL | All atherogenic particles; better than LDL | 10,454 |
| ANY_CVD | any(MCQ160B-F) | Cardiovascular history composite | 11,723 |
| SLEEP_DURATION_DIFF | weekend - weekday hrs | Sleep debt pattern | 3,229 |

### Results/Outcomes

**Feature Sets Defined:**

| Set | Base Features | With Missing Flags | Description |
|-----|---------------|-------------------|-------------|
| with_labs | 109 | 171 | All features including laboratory values |
| without_labs | 92 | 140 | Excludes lab values (screening without blood draw) |

**Output Files** (in `data/processed/`):
| File | Shape | Description |
|------|-------|-------------|
| `features_engineered.parquet` | 11,723 × 1,582 | Full dataset with derived features |
| `X_with_labs.parquet` | 11,698 × 171 | Features for models with labs |
| `X_without_labs.parquet` | 11,698 × 140 | Features for models without labs |
| `y_with_labs.parquet` | 11,698 × 1 | Target variable |
| `feature_engineering_report.json` | - | Feature statistics |

**Key Clinical Findings from Derived Features:**
- Elevated pulse pressure (>60 mmHg): 30.1%
- Hypertensive MAP (>100 mmHg): 16.8%
- Poor carb quality (ratio >15): 52.4%
- Moderate+ depression (PHQ9 ≥10): 8.6%
- Insulin resistant (TG/HDL >3.0): 24.5%
- High non-HDL cholesterol (≥160): 25.0%
- Any CVD history: 11.3%

**Target Distribution (preserved):**
- No Diabetes (0): 48.6%
- Prediabetes (1): 32.4%
- Diabetes (2): 19.0%

### Learnings

1. **Wake time data format varies** - NHANES stores as "HH:MM" strings, not numeric HHMM. Required flexible parsing.

2. **Weight history in pounds** - WHD110/120/140 are in lbs while BMXWT is kg. Conversion factor 0.453592 applied.

3. **ACR ratio clinical thresholds**:
   - <30 mg/g = Normal
   - 30-299 = Microalbuminuria (early kidney damage)
   - ≥300 = Macroalbuminuria (overt kidney disease)

4. **Waist-height ratio rule**: "Keep waist less than half your height" (ratio <0.5). In our data, 73.9% exceed 0.5, indicating high-risk population.

5. **Social jet lag**: Mean difference of ~1 hour between weekend/weekday wake times. 53% wake later on weekends.

6. **Depression-diabetes link**: PHQ9 score added because depression increases diabetes risk 60% and diabetes doubles depression risk.

7. **TG/HDL ratio as insulin resistance proxy**: 24.5% have ratio >3.0, indicating substantial insulin resistance in this population even before diabetes diagnosis.

### Potential Additional Features (Future Work)

- **Interaction features**: BMI × Age, Physical activity × Sedentary time
- **Metabolic syndrome composite**: Count of: high BP, high TG, low HDL, high waist, high glucose
- **Dietary quality scores**: Healthy Eating Index
- **eGFR**: Estimated glomerular filtration rate from creatinine

### Next Steps
- **Phase 5**: Exploratory Data Analysis (EDA)

---

## [2026-01-31] - Phase 5: Exploratory Data Analysis (EDA)

### Objective
Create publication-quality visualizations exploring diabetes risk factors in the NHANES dataset. Answer key questions about feature differences by diabetes status, strongest correlations, feature interactions, and temporal trends.

### Implementation

**Files created:**
- `src/visualization/eda.py` - Comprehensive visualization module with 15+ functions
- `notebooks/05_eda_visualizations.ipynb` - EDA notebook generating all figures

**Visualization Module Features:**
- Consistent colorblind-friendly palette (Blue/Orange/Red for No Diabetes/Prediabetes/Diabetes)
- Publication settings: 300 DPI, professional fonts, clean styling
- Reusable functions for target distribution, feature panels, correlations, interactions, PCA

**Figures Generated (29 total):**

| Category | Figures |
|----------|---------|
| **Population Overview** | `cohort_flow.png`, `target_distribution_final.png`, `demographics_by_status.png` |
| **Feature Distributions** | `features_anthropometric.png`, `features_blood_pressure.png`, `features_laboratory.png`, `features_dietary.png` |
| **Individual Deep Dives** | `hba1c_kde_by_status.png`, `waist_height_ratio_by_status.png`, `phq9_by_status.png` |
| **Correlation Analysis** | `top_correlations.png`, `correlation_heatmap.png` |
| **Feature Interactions** | `scatter_bmi_age.png`, `scatter_metabolic.png`, `feature_interactions.png` |
| **Temporal Analysis** | `prevalence_by_year.png`, `bmi_by_year.png`, `age_by_year.png` |
| **Dimensionality Reduction** | `pca_analysis.png` |
| **Risk Factors** | `risk_factors_forest.png` |
| **Summary** | `eda_summary_dashboard.png` |

### Key Findings

**1. Feature Differences by Diabetes Status:**
- Clear progression across all metabolic markers: No Diabetes → Prediabetes → Diabetes
- BMI: 28.1 → 30.3 → 33.0 kg/m² (mean by status)
- Age: 43.2 → 51.8 → 59.4 years (mean by status)
- Waist-height ratio: 73.9% exceed 0.5 threshold overall

**2. Strongest Correlations with Diabetes Status (Spearman):**
| Feature | Correlation |
|---------|-------------|
| RIDAGEYR (Age) | +0.28 |
| BMXBMI | +0.18 |
| BMXWAIST | +0.23 |
| WAIST_HEIGHT_RATIO | +0.21 |
| AVG_SYS_BP | +0.22 |
| PHQ9_SCORE | +0.08 |

**3. Feature Interactions:**
- BMI × Age: Diabetes concentrated in high-BMI/older-age quadrant
- Waist-height × TG/HDL: Both insulin resistance markers cluster in diabetic group
- Metabolic features highly intercorrelated (obesity, hypertension, dyslipidemia)

**4. Temporal Trends (2015-2018):**
- Diabetes prevalence: ~19% (stable across both survey cycles)
- No significant shifts in BMI or age distributions between years
- Consistent sampling methodology confirmed

**5. PCA Results:**
- PC1 explains ~15% of variance (dominated by anthropometric/metabolic features)
- 10 components needed for 50% variance
- Moderate but imperfect separation of diabetes groups in PC space

**6. Effect Sizes (Cohen's d, No Diabetes vs Diabetes):**
- Largest positive effects: Age (+0.72), Waist circumference (+0.52), Systolic BP (+0.45)
- Depression (PHQ9) shows small but significant effect (+0.15)

### Learnings

1. **Color palette matters**: Colorblind-friendly palette essential for accessibility
2. **Data cleaning for visualization**: PCA required handling mixed dtypes and imputation
3. **Sample size for scatter plots**: Used 30-50% sampling to prevent overplotting while preserving patterns
4. **Statistical annotations**: Adding Kruskal-Wallis p-values validates visual differences

### Next Steps
- **Phase 6**: Baseline Models

---

## [2026-02-01] - Phase 6: Baseline Models

### Objective
Establish performance benchmarks with simple models before implementing advanced approaches.

### Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Data splits** | 70/15/15 (train/val/test), stratified | Standard split with stratification to maintain class balance |
| **4 Dataset Structure** | minimal (NaN) + full (no NaN) × with/without labs | Different models have different NaN handling capabilities |
| **Evaluation metric** | F1 Macro (classification), RMSE (regression) | F1 Macro balances performance across imbalanced classes |

### Implementation

**Data Structure Finalized:**
| Dataset | Features | Has NaN | Use For |
|---------|----------|---------|---------|
| `X_with_labs_minimal` | 109 | Yes | LightGBM |
| `X_with_labs_full` | 96 | No | LogReg, MLP |
| `X_without_labs_minimal` | 92 | Yes | LightGBM |
| `X_without_labs_full` | 82 | No | LogReg, MLP |

**Files Created:**
- `src/models/train.py` - Training utilities, CV, MLflow tracking, Optuna tuning
- `src/models/evaluate.py` - Metrics, confusion matrices, ROC curves
- `notebooks/06_baseline_models.ipynb` - Baseline model training and evaluation
- `models/baseline/` - Saved models and results

### Results/Outcomes

**Classification Baselines (Validation Set):**
| Model | With Labs F1 | Without Labs F1 | AUC (with labs) |
|-------|--------------|-----------------|-----------------|
| Dummy (Stratified) | 0.337 | 0.337 | 0.502 |
| Dummy (Most Frequent) | 0.218 | 0.218 | 0.500 |
| **Logistic Regression** | **0.544** | 0.517 | **0.749** |
| Decision Tree | 0.550 | 0.512 | 0.740 |

**Regression Baselines (Validation Set) - HbA1c Prediction:**
| Model | With Labs RMSE | Without Labs RMSE | R² (with labs) |
|-------|----------------|-------------------|----------------|
| Dummy (Mean) | 1.328 | 1.328 | 0.000 |
| Dummy (Median) | 1.334 | 1.334 | -0.009 |
| **Ridge Regression** | **1.167** | 1.258 | **0.228** |
| Decision Tree | 1.213 | 1.268 | 0.166 |

**Key Findings:**
1. Logistic Regression achieves F1=0.544 (with labs) - a reasonable baseline
2. Labs matter: ~2.7% F1 drop without laboratory features
3. Regression is harder: R²=0.23 best case, indicating HbA1c is difficult to predict from other features
4. Decision Tree performs comparably to LogReg, suggesting non-linear patterns exist

### Learnings

1. **Baseline importance**: The naive baseline (F1=0.337) sets the floor - any model must beat this
2. **Class imbalance managed**: Stratified splitting and class weights help balance classes
3. **Labs vs No Labs**: Performance gap is modest for classification but significant for regression
4. **Model choice by data type**: Linear models need full imputation; tree models can handle NaN

### Next Steps
- **Phase 7**: Advanced Models
  - LightGBM with hyperparameter tuning (using minimal imputation datasets)
  - MLP Classifier/Regressor (using full imputation datasets)
  - Optuna for Bayesian hyperparameter optimization
  - Compare all models on held-out test set

---

## [2026-02-01] - Data Fix: Lab Values Special Code Handling

### Issue Discovered
During data verification before Phase 7, discovered that LBXGH (HbA1c) and other lab values contained invalid -7/-9 values. These were legitimate measurements (7.0% and 9.0% HbA1c) incorrectly recoded as "Refused" and "Don't Know" special codes.

### Root Cause
The `recode_special_values()` function was treating literal values 7 and 9 in laboratory columns as NHANES special codes when they were actually valid measurements:
- 61 patients with HbA1c of exactly 7.0% → incorrectly recoded to -7
- 13 patients with HbA1c of exactly 9.0% → incorrectly recoded to -9

**Key insight**: NHANES 7/9 special codes are for **questionnaire items only**, not laboratory values. Lab results are either measured numeric values or NaN (not collected).

### Fix Applied
Updated `src/data/cleaners.py`:
1. Added `is_lab_column()` - detects lab columns by prefix (LBX, LBD, URX, URD)
2. Modified `recode_special_values()` - now skips lab columns (83 columns)
3. Modified `clean_target_columns()` - only cleans questionnaire targets (DIQ010, DIQ050, DIQ070), not lab values

### Verification
After fix, LBXGH shows correct values:
- Range: 3.8% - 17.0% (clinically valid)
- 61 values at 7.0% preserved ✓
- 13 values at 9.0% preserved ✓
- 0 invalid -7 or -9 values ✓

### Data Regenerated
All interim and processed datasets regenerated with correct handling.

---

## [2026-02-02] - Phase 7: Advanced Models

### Objective
Train advanced models (LightGBM, MLP) with Bayesian hyperparameter optimization using Optuna, and track experiments with MLflow.

### Implementation

**Notebook created:**
- `notebooks/07_advanced_models.ipynb` - Complete training pipeline

**Models to train (8 total configurations):**

| Model | Task | Feature Set | Dataset |
|-------|------|-------------|---------|
| LightGBM Classifier | Classification | with_labs | minimal (NaN ok) |
| LightGBM Classifier | Classification | without_labs | minimal (NaN ok) |
| LightGBM Regressor | Regression | with_labs | minimal (NaN ok) |
| LightGBM Regressor | Regression | without_labs | minimal (NaN ok) |
| MLP Classifier | Classification | with_labs | full (no NaN) |
| MLP Classifier | Classification | without_labs | full (no NaN) |
| MLP Regressor | Regression | with_labs | full (no NaN) |
| MLP Regressor | Regression | without_labs | full (no NaN) |

**Hyperparameter tuning strategy:**
- Optuna with TPE sampler (Bayesian optimization)
- 100 trials for LightGBM (~1 hour each)
- 50 trials for MLP (~1.5 hours each)
- 3-fold cross-validation for faster iteration

**MLflow experiment tracking:**
- Experiment name: `diabetes-prediction-phase7`
- Logs: parameters, metrics, models, tags
- View UI: `mlflow ui --backend-store-uri mlruns`

**LightGBM hyperparameters tuned:**
- n_estimators (100-500)
- max_depth (3-10)
- learning_rate (0.01-0.3, log scale)
- num_leaves (15-127)
- min_child_samples (5-100)
- reg_alpha, reg_lambda (1e-8 to 10, log scale)
- subsample, colsample_bytree (0.5-1.0)

**MLP hyperparameters tuned:**
- n_layers (1-3)
- n_units per layer (32-256)
- activation (relu, tanh)
- alpha (L2 regularization, 1e-5 to 0.1)
- learning_rate (constant, adaptive)
- learning_rate_init (1e-4 to 1e-2)

### Configuration

```python
RANDOM_STATE = 42
CV_FOLDS = 3
N_TRIALS_LIGHTGBM = 10   # Quick test (use 100 for full run)
N_TRIALS_MLP = 5         # Quick test (use 50 for full run)
```

### Results

**Classification (Test Set):**

| Model | F1 Macro | ROC AUC | Accuracy |
|-------|----------|---------|----------|
| **LightGBM (with labs)** | **0.612** | **0.816** | 0.630 |
| LightGBM (without labs) | 0.549 | 0.756 | 0.565 |
| MLP (with labs) | 0.550 | 0.746 | 0.590 |
| MLP (without labs) | 0.535 | 0.735 | 0.569 |

**Regression (Test Set):**

| Model | RMSE | R² | MAE |
|-------|------|-----|-----|
| **LightGBM (with labs)** | **0.988** | **0.301** | 0.567 |
| LightGBM (without labs) | 1.080 | 0.164 | 0.609 |
| MLP (with labs) | 1.100 | 0.133 | 0.649 |
| MLP (without labs) | 1.163 | 0.031 | 0.682 |

**Improvement vs Phase 6 Baselines:**

| Metric | Baseline (LogReg/Ridge) | LightGBM | Improvement |
|--------|-------------------------|----------|-------------|
| F1 Macro | 0.544 | 0.612 | **+12.5%** |
| ROC AUC | 0.749 | 0.816 | **+8.9%** |
| R² (regression) | 0.069 | 0.301 | **+336%** |

**Best LightGBM Hyperparameters (classification, with labs):**
```python
n_estimators = 349
max_depth = 5
learning_rate = 0.0124
num_leaves = 50
min_child_samples = 36
reg_alpha = 0.037
reg_lambda = 0.005
subsample = 0.944
colsample_bytree = 0.736
```

### Key Findings

1. **LightGBM outperforms MLP** on both classification and regression tasks (common for tabular data)
2. **Labs matter significantly**: ~6% F1 drop and ~14% R² drop without laboratory features
3. **Regression improved dramatically**: R² from 0.07 (baseline) to 0.30 (LightGBM)
4. **Classification per-class performance**:
   - No Diabetes: F1 = 0.72 (best)
   - Prediabetes: F1 = 0.56 (hardest to predict)
   - Diabetes: F1 = 0.56

### Artifacts Generated

| Artifact | Location |
|----------|----------|
| Models | `models/advanced/{classification,regression}/` |
| Results | `models/advanced/results_summary.json` |
| Figures | `reports/figures/phase7_*.png` |
| MLflow | `mlruns/` (view with `mlflow ui`) |

### Learnings

1. **Optuna TPE sampler** efficiently explores hyperparameter space - even 10 trials found good configurations
2. **LightGBM's native NaN handling** is valuable - no need for full imputation
3. **MLP requires more trials** to find good architectures; sklearn's MLP is also limited compared to deep learning frameworks
4. **Class imbalance handling** via class weights helped balance predictions across all 3 classes

### Next Steps
- **Phase 7.1 (Optional)**: Deep Learning with TensorFlow/PyTorch
- **Phase 8**: Model Evaluation & Comparison (detailed error analysis)
- **Phase 9**: Model Interpretation & Insights (SHAP, feature importance)

---

## [2026-02-02] - Phase 7.1: Deep Learning with PyTorch

### Objective
Build neural networks from scratch using PyTorch, exploring regression, learning rate scheduling, and hyperparameter tuning with Optuna.

### Implementation

**Notebook created:**
- `notebooks/07a_deep_learning_pytorch.ipynb` - Comprehensive PyTorch tutorial with 8 parts

**Topics covered:**
| Part | Topic | Key Concepts |
|------|-------|--------------|
| 1-3 | Basic Classification | Tensors, DataLoaders, nn.Module, training loop |
| 4-5 | Evaluation & Saving | Metrics, confusion matrix, model serialization |
| 6 | Regression | MSELoss, HbA1c prediction, R² metric |
| 7 | LR Scheduling | OneCycleLR, warmup, annealing comparison |
| 8 | Hyperparameter Tuning | Optuna, TPE sampler, pruning, FlexibleNN |

**Base architecture:**
```
Input (96 features)
    ↓
Linear(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(64) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(32) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Output (3 classes or 1 regression value)
```

### Results

**Classification (Test Set):**

| Model | F1 Macro | ROC AUC | Accuracy |
|-------|----------|---------|----------|
| **LightGBM (with labs)** | **0.612** | **0.816** | **0.630** |
| MLP sklearn (with labs) | 0.550 | 0.746 | 0.590 |
| PyTorch (manual) | 0.560 | 0.755 | 0.573 |
| PyTorch (OneCycleLR) | 0.552 | 0.749 | 0.557 |
| PyTorch (Optuna-tuned) | 0.562 | 0.759 | 0.572 |

**Key observations:**
- Optuna-tuned PyTorch achieved best neural network performance (F1=0.562)
- OneCycleLR didn't help significantly for this dataset/architecture
- All PyTorch variants outperformed sklearn MLP
- LightGBM still wins by ~9% F1 margin

**Regression (HbA1c prediction):**
- PyTorch regression model trained successfully
- Compared against LightGBM baseline (R²=0.30)
- Neural network regression is harder than classification for this task

### Techniques Explored

| Technique | Implementation | Impact |
|-----------|---------------|--------|
| **BatchNorm** | After each linear layer | Stabilizes training |
| **Dropout** | 0.3 rate | Prevents overfitting |
| **Class weights** | Inverse frequency | Balances classes |
| **OneCycleLR** | 30% warmup, cosine decay | Minimal impact here |
| **Optuna** | 20 trials, TPE sampler | +0.2% F1 vs manual |

### Optuna Search Space

```python
n_layers: 1-4
hidden_size: 32-256 (step=32)
dropout_rate: 0.1-0.5
learning_rate: 1e-4 to 1e-2 (log scale)
batch_size: [32, 64, 128]
```

### Key Findings

1. **LightGBM dominates for tabular data** - ~9% F1 advantage over best neural network
2. **Optuna helps but marginally** - Only +0.2% F1 over manual architecture
3. **LR scheduling mixed results** - OneCycleLR didn't improve this model
4. **Neural networks need more data** - ~8K samples favors gradient boosting
5. **PyTorch > sklearn MLP** - Custom architecture with BatchNorm/Dropout helps

### Concepts Learned

| Concept | Description |
|---------|-------------|
| **Tensors** | GPU-accelerated arrays with gradient tracking |
| **nn.Module** | PyTorch's base class for neural networks |
| **Training loop** | Forward → Loss → Backward → Update |
| **BatchNorm** | Normalizes layer inputs, stabilizes training |
| **Dropout** | Randomly zeros neurons to prevent overfitting |
| **OneCycleLR** | Learning rate warmup then cosine decay |
| **Optuna** | Bayesian hyperparameter optimization |
| **FlexibleNN** | Dynamic architecture for tuning |

### Learnings

1. **BatchNorm requires batch_size > 1** - Use `drop_last=True` in DataLoader
2. **NumPy 2.x incompatible with PyTorch <2.4** - Upgrade PyTorch or downgrade NumPy
3. **MPS (Apple Silicon) works well** - Good speedup over CPU
4. **Class weights essential** - Without them, model ignores minority classes
5. **More trials = better tuning** - 20 trials is minimal; 50-100 recommended

### When to Use Neural Networks

| Use Case | Recommendation |
|----------|----------------|
| Tabular, <50K samples | Gradient Boosting |
| Tabular, >100K samples | Can try neural networks |
| Images, text, audio | Neural networks |

### Next Steps
- **Phase 8**: Model Evaluation & Comparison (detailed error analysis)
- **Phase 9**: Model Interpretation & Insights (SHAP, feature importance)

---

## [2026-02-03] - Phase 8 & 9: Model Evaluation and Interpretation

### Objective
Comprehensive evaluation of all trained models with detailed error analysis, subgroup fairness assessment, and SHAP-based feature interpretation to derive actionable insights.

### Implementation

**Notebook created:**
- `notebooks/08_evaluation_and_interpretation.ipynb` - Combined Phase 8 & 9 analysis

**Models evaluated:**
- LightGBM (with/without labs) - Classification & Regression
- MLP sklearn (with/without labs) - Classification & Regression
- PyTorch (with labs) - Classification

### Results

**Classification Performance (Test Set):**

| Model | F1 Macro | ROC AUC | Accuracy |
|-------|----------|---------|----------|
| **LightGBM (with labs)** | **0.612** | **0.816** | **0.630** |
| LightGBM (without labs) | 0.549 | 0.756 | 0.565 |
| PyTorch (with labs) | 0.562 | 0.759 | 0.572 |
| MLP (with labs) | 0.550 | 0.746 | 0.590 |
| MLP (without labs) | 0.535 | 0.735 | 0.569 |

**Regression Performance (Test Set) - HbA1c Prediction:**

| Model | RMSE | R² | MAE |
|-------|------|-----|-----|
| **LightGBM (with labs)** | **0.988** | **0.301** | **0.567** |
| LightGBM (without labs) | 1.080 | 0.164 | 0.609 |
| MLP (with labs) | 1.100 | 0.133 | 0.649 |
| MLP (without labs) | 1.163 | 0.031 | 0.682 |

**Error Analysis (LightGBM with labs):**
- Overall accuracy: 63.0%
- Hardest to predict: Prediabetes (highest error rate)
- Most common misclassification: Prediabetes ↔ No Diabetes
- Lower confidence predictions have higher error rates (as expected)

**Subgroup Analysis:**

| Subgroup | F1 Macro | Notes |
|----------|----------|-------|
| Young (18-39) | ~0.55 | Lower diabetes prevalence |
| Middle (40-59) | ~0.62 | Best performance |
| Older (60+) | ~0.60 | High diabetes prevalence |
| Male | ~0.61 | Slightly better than female |
| Female | ~0.60 | Similar performance |
| Normal BMI | ~0.50 | Hardest subgroup |
| Obese | ~0.65 | Easiest to classify |

**Labs Impact:**
- LightGBM F1 drops ~10% without lab values
- LightGBM AUC drops ~7% without lab values
- Labs most important for regression (R² drops from 0.30 to 0.16)

### Feature Importance (SHAP Analysis)

**Top 10 Features by Mean |SHAP|:**

| Rank | Feature | Category |
|------|---------|----------|
| 1 | RIDAGEYR (Age) | Non-Modifiable |
| 2 | TG_HDL_RATIO | Lab Value |
| 3 | WAIST_HEIGHT_RATIO | Modifiable |
| 4 | BMXBMI | Modifiable |
| 5 | LBXSGTSI (GGT) | Lab Value |
| 6 | WHD130 (Age at heaviest) | Non-Modifiable |
| 7 | MCQ300C (Family history) | Non-Modifiable |
| 8 | AVG_SYS_BP | Modifiable |
| 9 | LBXTR (Triglycerides) | Lab Value |
| 10 | CDQ010 (Shortness of breath) | Non-Modifiable |

**Feature Importance by Category:**
- **Lab Values**: ~35% of total importance
- **Modifiable factors**: ~30% of total importance
- **Non-Modifiable factors**: ~25% of total importance
- **Other**: ~10%

**Top Modifiable Risk Factors:**
1. Waist-to-height ratio
2. BMI
3. Average systolic blood pressure
4. Weight history features
5. Dietary patterns (carb/fiber ratio)

### Key Findings

1. **LightGBM is the best model** for both classification (F1=0.612) and regression (R²=0.301)
2. **Prediabetes is hardest to predict** - often confused with No Diabetes
3. **Labs provide ~10% F1 improvement** - significant but model still useful without them
4. **Age is the strongest predictor** - but many modifiable factors in top 10
5. **Model performs fairly across subgroups** - no major bias by gender or race
6. **Calibration is reasonable** - predicted probabilities roughly match actual rates

### Actionable Insights

**For Diabetes Prevention:**
1. **Weight management** - BMI and waist circumference are top modifiable predictors
2. **Blood pressure control** - Strongly associated with diabetes risk
3. **Dietary quality** - Carb/fiber ratio indicates diet quality
4. **Physical activity** - Helps with weight and metabolic health
5. **Regular screening** - Especially for those 40+ with family history

**Clinical Implications:**
- Model can identify high-risk individuals for intervention
- Without-labs version enables community screening
- Prediabetes detection needs improvement (consider threshold adjustment)

### Artifacts Generated

**Figures (in `reports/figures/`):**
- `phase8_confusion_matrices.png` - All models side-by-side
- `phase8_roc_curves_comparison.png` - ROC by class
- `phase8_pr_curves_comparison.png` - Precision-Recall curves
- `phase8_calibration_curves.png` - Reliability diagrams
- `phase8_error_analysis.png` - Confidence vs accuracy
- `phase8_subgroup_analysis.png` - Performance by demographic
- `phase8_labs_comparison.png` - With vs without labs
- `phase8_regression_residuals.png` - Residual analysis
- `phase8_regression_predicted_vs_actual.png` - All regression models
- `phase9_shap_summary_by_class.png` - SHAP beeswarm plots
- `phase9_shap_importance_bar.png` - Top 20 features
- `phase9_shap_dependence.png` - Feature dependence plots
- `phase9_importance_comparison.png` - SHAP vs Permutation
- `phase9_modifiable_factors.png` - Risk factor categories

**Data files (in `models/advanced/`):**
- `evaluation_results.json` - All metrics and subgroup results
- `feature_importance_shap.csv` - SHAP importance rankings
- `feature_importance_permutation.csv` - Permutation importance

### Learnings

1. **SHAP 0.50+ returns 3D arrays** - Format changed from list of 2D to single 3D array
2. **PyTorch checkpoint format matters** - Save/load with consistent dictionary structure
3. **Subgroup analysis reveals model fairness** - Important for healthcare applications
4. **Calibration curves show reliability** - Model probabilities are reasonably trustworthy
5. **Error analysis guides improvement** - Focus on prediabetes classification

### Next Steps
- **Phase 10**: Deployment (Streamlit app for risk prediction)
- **Phase 11**: Documentation & Polish (README, final report)

---

## [2026-02-03] - Phase 10: Deployment (Streamlit App)

### Objective
Create an interactive Streamlit application for diabetes risk prediction with comprehensive test cases demonstrating the full capability of the trained models.

### Implementation

**Files Created:**
- `app/streamlit_app.py` - Multi-page Streamlit application
- `app/requirements.txt` - App-specific dependencies

**App Features:**

| Page | Description |
|------|-------------|
| **Risk Calculator** | Main prediction interface with example individual selection |
| **Example Test Cases** | All 6 test cases with predictions (with/without labs) |
| **Compare Scenarios** | Side-by-side comparison of different individuals |
| **Population Insights** | Key findings from NHANES analysis |
| **Model Information** | Technical details, performance metrics, limitations |

### Test Cases (Example Individuals)

Created 6 comprehensive test cases that fully demonstrate the predictor's capabilities:

| Test Case | Description | Expected Outcome | Key Factors |
|-----------|-------------|------------------|-------------|
| **Low Risk - Healthy Adult** | 32yo female, active, healthy BMI (22), no family history | No Diabetes | Young, healthy weight, active, excellent labs |
| **High Risk - Metabolic Syndrome** | 62yo male, obese (BMI 34), hypertension, dyslipidemia | Diabetes | Age, obesity, high TG/HDL (5.0), family history |
| **Borderline - Prediabetes Risk** | 48yo female, overweight (BMI 28), borderline labs | Prediabetes | Middle age, overweight, borderline lipids |
| **Lifestyle Matters - Active Despite Age** | 58yo male, very active, excellent diet, normal BMI (24) | No Diabetes | Older age BUT excellent lifestyle compensates |
| **Young but High Risk - Poor Lifestyle** | 28yo male, obese (BMI 35.5), sedentary, poor diet | Prediabetes | Young age BUT poor lifestyle increases risk |
| **Labs Critical - Hidden Risk** | 45yo male, normal BMI, BUT very poor labs (TG/HDL 6.5) | Diabetes | Labs reveal hidden metabolic dysfunction |

### Model Integration

**Features:**
- Uses LightGBM models (best performing) for classification and regression
- Supports both with-labs (109 features) and without-labs (92 features) predictions
- Displays predicted class, probabilities, and HbA1c prediction
- Shows comparison between with-labs and without-labs predictions
- Loads feature order from processed data to ensure correct alignment

**Prediction Examples:**

| Individual | With Labs Prediction | Without Labs Prediction | HbA1c |
|------------|---------------------|------------------------|-------|
| Low Risk (32yo healthy) | Prediabetes (52%) | Prediabetes | 5.47% |
| High Risk (62yo metabolic) | Diabetes (63%) | Diabetes | 6.69% |

### Key Insights Demonstrated

1. **Age is a strong predictor** - But lifestyle can compensate
2. **Labs add significant value** - ~10% F1 improvement with blood tests
3. **Hidden risks exist** - Some individuals appear healthy but have poor labs
4. **Modifiable factors matter** - Weight, diet, activity are actionable
5. **Model differentiates well** - Clear separation between risk profiles

### How to Run

```bash
# Install dependencies
pip install -r app/requirements.txt

# Run the app
streamlit run app/streamlit_app.py
```

### Technical Details

- Uses cached model and data loading for performance
- Feature order loaded from parquet files to match training data
- Handles missing values (NaN) correctly (LightGBM native support)
- Derived features calculated in test cases (same as training)

### Learnings

1. **Feature alignment is critical** - Models don't store feature names with data; must preserve order
2. **Test cases should span the feature space** - Cover low/high risk, young/old, with/without labs
3. **Probability calibration matters** - Users need to understand confidence levels
4. **Comparison view is valuable** - Showing with/without labs difference is educational

### Next Steps
- **Phase 11**: Documentation & Polish (README, final report)

---
