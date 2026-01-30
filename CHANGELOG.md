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
| File | Shape | Description |
|------|-------|-------------|
| `features_engineered.parquet` | 11,723 × 1,582 | All features + 20 derived |
| `X_with_labs.parquet` | 11,698 × 171 | Modeling features (with labs) |
| `X_without_labs.parquet` | 11,698 × 140 | Modeling features (no labs) |
| `y_with_labs.parquet` | 11,698 × 1 | Target variable |

**Key Constants** (defined in `src/data/cleaners.py`):
```python
TARGET_RELATED_COLS = {'LBXGH', 'LBXGLU', 'DIQ010', 'DIQ050', 'DIQ070', 'DIABETES_STATUS'}
# These are NEVER imputed - they define the target variable

MISSING_FLAG_THRESHOLD = 0.05   # Create _MISSING flags for ≥5% missing
IMPUTE_THRESHOLD = 0.05         # Minimal imputation: only <5% missing
MAX_MISSING_RATE = 0.50         # Full imputation: only ≤50% missing
```

**Special Value Encoding**:
- `-7` = Refused (originally 7, 77, 777...)
- `-9` = Don't Know (originally 9, 99, 999...)
- `NaN` = Not collected / structurally missing

**Feature Set** (defined in PRD section 4.2):
- ~95 base features + ~10 derived features
- 11 features have >50% missing in 2015-2018 (marked † in PRD) - may improve with full dataset
- Option to add more features from raw dataset later

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
  - Univariate/bivariate analysis
  - Feature correlations
  - Publication-quality visualizations

---
