# Diabetes Predictor

A machine learning project to predict diabetes risk using NHANES data (1999-2018).

## Description

This project uses machine learning algorithms to predict the likelihood of diabetes based on health metrics from the National Health and Nutrition Examination Survey (NHANES).

## Setup

### 1. Create and activate conda environment

```bash
conda env create -f environment.yml
conda activate diabetes-ml
```

Or if the environment already exists:

```bash
conda activate diabetes-ml
pip install -r requirements.txt
```

### 2. Download NHANES data

```bash
python -m src.data.download --years 2015-2016 2017-2018
```

### 3. Verify installation

```bash
python -c "import pandas, sklearn, mlflow; print('All dependencies installed')"
```

## Project Structure

```
diabetes_prediction_project/
├── config/              # Configuration files
├── data/                # Data files (raw, interim, processed)
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
├── models/              # Saved model artifacts
├── reports/             # Generated reports and figures
└── tests/               # Unit tests
```

## Current Progress

See [CHANGELOG.md](CHANGELOG.md) for detailed progress and decisions.
