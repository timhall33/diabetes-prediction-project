"""
NHANES Data Cleaning Utilities

Functions to clean and preprocess NHANES data for modeling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# NHANES special value patterns
REFUSED_PATTERNS = {7, 77, 777, 7777, 77777}
DONT_KNOW_PATTERNS = {9, 99, 999, 9999, 99999}

# Target-related columns that should NEVER be imputed
TARGET_RELATED_COLS = {
    'LBXGH',      # HbA1c - regression target
    'LBXGLU',     # Fasting glucose - classification target component
    'DIQ010',     # Self-reported diabetes
    'DIQ050',     # Taking diabetes pills
    'DIQ070',     # Taking insulin
    'DIABETES_STATUS',  # Derived target variable
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_config() -> dict:
    """Load the project configuration."""
    config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def recode_special_values(
    df: pd.DataFrame,
    refused_code: int = -7,
    dont_know_code: int = -9,
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Recode NHANES special values to distinct codes.

    NHANES uses patterns like 7, 77, 777 for "Refused" and 9, 99, 999 for "Don't know".
    This function recodes them to preserve the semantic meaning.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    refused_code : int, default -7
        Code to use for "Refused" responses.
    dont_know_code : int, default -9
        Code to use for "Don't know" responses.
    columns : list, optional
        Specific columns to process. If None, processes all numeric columns.

    Returns
    -------
    tuple
        (cleaned_df, changes_log) where changes_log documents what was changed.
    """
    df = df.copy()
    changes_log = {'refused': {}, 'dont_know': {}}

    if columns is None:
        # Only process numeric columns (special values are numeric)
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        # Count refused values
        refused_mask = df[col].isin(REFUSED_PATTERNS)
        refused_count = refused_mask.sum()
        if refused_count > 0:
            df.loc[refused_mask, col] = refused_code
            changes_log['refused'][col] = int(refused_count)

        # Count don't know values
        dont_know_mask = df[col].isin(DONT_KNOW_PATTERNS)
        dont_know_count = dont_know_mask.sum()
        if dont_know_count > 0:
            df.loc[dont_know_mask, col] = dont_know_code
            changes_log['dont_know'][col] = int(dont_know_count)

    total_refused = sum(changes_log['refused'].values())
    total_dont_know = sum(changes_log['dont_know'].values())
    logger.info(f"Recoded {total_refused} 'Refused' values to {refused_code}")
    logger.info(f"Recoded {total_dont_know} 'Don't know' values to {dont_know_code}")

    return df, changes_log


def create_missing_flags(
    df: pd.DataFrame,
    threshold: float = 0.05,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add _MISSING indicator columns for features with >= threshold missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    threshold : float, default 0.05
        Minimum missing rate to create a flag (0.05 = 5%).
    exclude_cols : list, optional
        Columns to exclude from flag creation.

    Returns
    -------
    tuple
        (df_with_flags, flag_columns) where flag_columns lists the new columns.
    """
    df = df.copy()
    exclude_cols = set(exclude_cols or [])
    exclude_cols.update(['SEQN', 'SURVEY_YEAR'])  # Always exclude identifiers

    flag_columns = []
    missing_rates = df.isna().mean()

    for col in df.columns:
        if col in exclude_cols:
            continue
        if col.endswith('_MISSING'):  # Don't flag flags
            continue

        missing_rate = missing_rates[col]
        if missing_rate >= threshold:
            flag_col = f"{col}_MISSING"
            df[flag_col] = df[col].isna().astype(int)
            flag_columns.append(flag_col)

    logger.info(f"Created {len(flag_columns)} missing indicator columns (threshold={threshold*100:.0f}%)")

    return df, flag_columns


def get_imputation_values(
    df: pd.DataFrame,
    columns: List[str]
) -> Dict[str, Union[float, int, str]]:
    """
    Calculate imputation values (median for numeric, mode for categorical).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list
        Columns to calculate imputation values for.

    Returns
    -------
    dict
        Mapping of column name to imputation value.
    """
    impute_values = {}

    for col in columns:
        if col not in df.columns:
            continue

        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Numeric: use median (robust to outliers)
            impute_values[col] = df[col].median()
        else:
            # Categorical: use mode
            mode_result = df[col].mode()
            if len(mode_result) > 0:
                impute_values[col] = mode_result.iloc[0]

    return impute_values


def impute_low_missing(
    df: pd.DataFrame,
    threshold: float = 0.05,
    target_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute only features with < threshold missing rate.

    This is the "minimal imputation" strategy for tree-based models that
    can handle NaN natively. We only impute where it's very safe to do so.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    threshold : float, default 0.05
        Maximum missing rate to impute (0.05 = 5%).
    target_cols : list, optional
        Target-related columns to never impute.

    Returns
    -------
    tuple
        (imputed_df, imputation_log)
    """
    df = df.copy()
    target_cols = set(target_cols or [])
    target_cols.update(TARGET_RELATED_COLS)

    imputation_log = {}
    missing_rates = df.isna().mean()

    # Find columns to impute
    cols_to_impute = []
    for col in df.columns:
        if col in target_cols:
            continue
        if col.endswith('_MISSING'):  # Don't impute flags
            continue
        if col in ['SEQN', 'SURVEY_YEAR']:  # Don't impute identifiers
            continue

        missing_rate = missing_rates[col]
        if 0 < missing_rate < threshold:
            cols_to_impute.append(col)

    # Calculate and apply imputation values
    impute_values = get_imputation_values(df, cols_to_impute)

    for col, value in impute_values.items():
        n_missing = df[col].isna().sum()
        df[col] = df[col].fillna(value)
        imputation_log[col] = {
            'n_imputed': int(n_missing),
            'impute_value': float(value) if isinstance(value, (int, float, np.number)) else str(value),
            'missing_rate': float(missing_rates[col])
        }

    logger.info(f"Imputed {len(imputation_log)} columns with <{threshold*100:.0f}% missing")

    return df, imputation_log


def impute_all_features(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    max_missing_rate: float = 0.50
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute features with missing values up to a maximum missing rate.

    This is for models that cannot handle NaN (logistic regression, neural networks).
    Features with >max_missing_rate are left as NaN (too unreliable to impute).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_cols : list, optional
        Target-related columns to never impute.
    max_missing_rate : float, default 0.50
        Maximum missing rate to impute. Features with higher missing rates
        are left as NaN and should be excluded from modeling.

    Returns
    -------
    tuple
        (imputed_df, imputation_log)
    """
    df = df.copy()
    target_cols = set(target_cols or [])
    target_cols.update(TARGET_RELATED_COLS)

    imputation_log = {}
    skipped_log = {}
    missing_rates = df.isna().mean()

    # Find columns to impute (respecting max_missing_rate threshold)
    cols_to_impute = []
    for col in df.columns:
        if col in target_cols:
            continue
        if col.endswith('_MISSING'):
            continue
        if col in ['SEQN', 'SURVEY_YEAR']:
            continue

        missing_rate = missing_rates[col]
        if missing_rate > 0 and missing_rate <= max_missing_rate:
            cols_to_impute.append(col)
        elif missing_rate > max_missing_rate:
            skipped_log[col] = float(missing_rate)

    # Calculate and apply imputation values
    impute_values = get_imputation_values(df, cols_to_impute)

    for col, value in impute_values.items():
        n_missing = df[col].isna().sum()
        df[col] = df[col].fillna(value)
        imputation_log[col] = {
            'n_imputed': int(n_missing),
            'impute_value': float(value) if isinstance(value, (int, float, np.number)) else str(value),
            'missing_rate': float(missing_rates[col])
        }

    logger.info(f"Imputed {len(imputation_log)} columns (missing rate <= {max_missing_rate*100:.0f}%)")
    if skipped_log:
        logger.info(f"Skipped {len(skipped_log)} columns with >{max_missing_rate*100:.0f}% missing (too unreliable)")

    return df, imputation_log


def clean_pipeline(
    df: pd.DataFrame,
    minimal_impute: bool = True,
    missing_flag_threshold: float = 0.05,
    impute_threshold: float = 0.05
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main cleaning pipeline.

    Steps:
    1. Recode special values (Refused → -7, Don't Know → -9)
    2. Create missing flags for features with >= threshold missing
    3. Impute features (minimal or full, depending on minimal_impute flag)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    minimal_impute : bool, default True
        If True, only impute <5% missing (for tree models).
        If False, impute all missing (for linear/NN models).
    missing_flag_threshold : float, default 0.05
        Threshold for creating _MISSING flag columns.
    impute_threshold : float, default 0.05
        Threshold for minimal imputation (only used if minimal_impute=True).

    Returns
    -------
    tuple
        (cleaned_df, cleaning_report)
    """
    logger.info("=" * 60)
    logger.info("Starting data cleaning pipeline")
    logger.info(f"  Input shape: {df.shape}")
    logger.info(f"  Minimal imputation: {minimal_impute}")
    logger.info("=" * 60)

    cleaning_report = {
        'input_shape': df.shape,
        'minimal_impute': minimal_impute,
        'steps': {}
    }

    # Step 1: Recode special values
    logger.info("\nStep 1: Recoding special values...")
    df, recode_log = recode_special_values(df)
    cleaning_report['steps']['recode_special_values'] = recode_log

    # Step 2: Create missing flags
    logger.info("\nStep 2: Creating missing flags...")
    df, flag_columns = create_missing_flags(df, threshold=missing_flag_threshold)
    cleaning_report['steps']['missing_flags'] = {
        'threshold': missing_flag_threshold,
        'n_flags_created': len(flag_columns),
        'flag_columns': flag_columns
    }

    # Step 3: Imputation
    logger.info("\nStep 3: Imputing missing values...")
    if minimal_impute:
        df, impute_log = impute_low_missing(df, threshold=impute_threshold)
    else:
        df, impute_log = impute_all_features(df)
    cleaning_report['steps']['imputation'] = {
        'strategy': 'minimal' if minimal_impute else 'full',
        'threshold': impute_threshold if minimal_impute else None,
        'columns_imputed': impute_log
    }

    # Final summary
    cleaning_report['output_shape'] = df.shape
    cleaning_report['n_missing_after'] = int(df.isna().sum().sum())

    logger.info("\n" + "=" * 60)
    logger.info("Cleaning pipeline complete")
    logger.info(f"  Output shape: {df.shape}")
    logger.info(f"  Total missing values remaining: {cleaning_report['n_missing_after']:,}")
    logger.info("=" * 60)

    return df, cleaning_report
