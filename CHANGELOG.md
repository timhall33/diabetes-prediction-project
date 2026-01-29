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
  - Load all downloaded files
  - Profile data structure and distributions
  - Analyze missing value patterns
  - Define and validate target variable

---
