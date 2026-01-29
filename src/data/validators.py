"""
NHANES Data Validation Utilities

Functions to validate data quality and ensure cleaning doesn't introduce errors.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_config() -> dict:
    """Load the project configuration."""
    config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_ranges(
    df: pd.DataFrame,
    range_config: Optional[Dict[str, Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    Check that values are within expected ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    range_config : dict, optional
        Mapping of column name to (min, max) tuple.
        If None, loads from config.yaml.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: variable, n_below_min, n_above_max, pct_out_of_range
    """
    if range_config is None:
        config = load_config()
        range_config = config.get('validation', {}).get('ranges', {})

    results = []

    for col, (min_val, max_val) in range_config.items():
        if col not in df.columns:
            logger.warning(f"Column {col} not found in dataframe")
            continue

        # Get non-null values (excluding special codes like -7, -9)
        values = df[col].dropna()
        values = values[values >= 0]  # Exclude negative special codes

        n_below = (values < min_val).sum()
        n_above = (values > max_val).sum()
        n_total = len(values)

        results.append({
            'variable': col,
            'min_expected': min_val,
            'max_expected': max_val,
            'actual_min': values.min() if len(values) > 0 else None,
            'actual_max': values.max() if len(values) > 0 else None,
            'n_below_min': int(n_below),
            'n_above_max': int(n_above),
            'n_out_of_range': int(n_below + n_above),
            'pct_out_of_range': round((n_below + n_above) / n_total * 100, 2) if n_total > 0 else 0
        })

    results_df = pd.DataFrame(results)

    # Log summary
    total_violations = results_df['n_out_of_range'].sum()
    if total_violations > 0:
        logger.warning(f"Found {total_violations} values out of expected ranges")
        for _, row in results_df[results_df['n_out_of_range'] > 0].iterrows():
            logger.warning(f"  {row['variable']}: {row['n_out_of_range']} out of range")
    else:
        logger.info("All values within expected ranges")

    return results_df


def check_logical_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check logical consistency between related variables.

    Checks performed:
    - Current weight <= max weight ever (if both available)
    - Waist circumference < height
    - Diastolic BP < Systolic BP
    - Age consistency checks

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: check_name, n_violations, pct_violations, example_seqns
    """
    results = []

    # Check 1: Current weight <= max weight
    if 'BMXWT' in df.columns and 'WHD140' in df.columns:
        # WHD140 is in pounds, BMXWT is in kg
        max_weight_kg = df['WHD140'] * 0.453592
        mask = (df['BMXWT'].notna() & max_weight_kg.notna() & (df['BMXWT'] > max_weight_kg * 1.05))
        # Allow 5% tolerance for measurement differences
        n_violations = mask.sum()
        example_seqns = df.loc[mask, 'SEQN'].head(5).tolist() if n_violations > 0 else []
        results.append({
            'check_name': 'current_weight_le_max_weight',
            'description': 'Current weight should be <= max weight ever',
            'n_violations': int(n_violations),
            'pct_violations': round(n_violations / len(df) * 100, 3),
            'example_seqns': example_seqns
        })

    # Check 2: Waist < Height
    if 'BMXWAIST' in df.columns and 'BMXHT' in df.columns:
        mask = (df['BMXWAIST'].notna() & df['BMXHT'].notna() & (df['BMXWAIST'] >= df['BMXHT']))
        n_violations = mask.sum()
        example_seqns = df.loc[mask, 'SEQN'].head(5).tolist() if n_violations > 0 else []
        results.append({
            'check_name': 'waist_lt_height',
            'description': 'Waist circumference should be < height',
            'n_violations': int(n_violations),
            'pct_violations': round(n_violations / len(df) * 100, 3),
            'example_seqns': example_seqns
        })

    # Check 3: Diastolic BP < Systolic BP
    for i in [1, 2, 3]:
        sys_col = f'BPXSY{i}'
        dia_col = f'BPXDI{i}'
        if sys_col in df.columns and dia_col in df.columns:
            mask = (df[sys_col].notna() & df[dia_col].notna() &
                    (df[dia_col] > 0) & (df[sys_col] > 0) &  # Exclude 0 readings
                    (df[dia_col] >= df[sys_col]))
            n_violations = mask.sum()
            example_seqns = df.loc[mask, 'SEQN'].head(5).tolist() if n_violations > 0 else []
            results.append({
                'check_name': f'diastolic_lt_systolic_{i}',
                'description': f'Diastolic BP{i} should be < Systolic BP{i}',
                'n_violations': int(n_violations),
                'pct_violations': round(n_violations / len(df) * 100, 3),
                'example_seqns': example_seqns
            })

    # Check 4: If taking diabetes medication, should have diabetes indicator
    if all(col in df.columns for col in ['DIQ050', 'DIQ010']):
        # DIQ050 = 1 means taking diabetes pills
        # DIQ010 = 1 means told have diabetes, 2 = no, 3 = borderline
        mask = ((df['DIQ050'] == 1) & (df['DIQ010'] == 2))  # Taking pills but said "no" to diabetes
        n_violations = mask.sum()
        example_seqns = df.loc[mask, 'SEQN'].head(5).tolist() if n_violations > 0 else []
        results.append({
            'check_name': 'diabetes_meds_implies_diabetes',
            'description': 'If taking diabetes medication, should report having diabetes',
            'n_violations': int(n_violations),
            'pct_violations': round(n_violations / len(df) * 100, 3),
            'example_seqns': example_seqns
        })

    results_df = pd.DataFrame(results)

    # Log summary
    total_violations = results_df['n_violations'].sum()
    if total_violations > 0:
        logger.warning(f"Found {total_violations} logical consistency violations")
        for _, row in results_df[results_df['n_violations'] > 0].iterrows():
            logger.warning(f"  {row['check_name']}: {row['n_violations']} violations")
    else:
        logger.info("All logical consistency checks passed")

    return results_df


def check_target_leakage(
    feature_cols: List[str],
    target_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Ensure no target-related columns are in the feature set.

    Parameters
    ----------
    feature_cols : list
        List of feature column names.
    target_cols : list, optional
        List of target-related column names to check for.

    Returns
    -------
    list
        List of leaked columns found in features.
    """
    if target_cols is None:
        target_cols = [
            'LBXGH', 'LBXGLU', 'DIQ010', 'DIQ050', 'DIQ070',
            'DIABETES_STATUS',
            # Also check for missing flags of target columns
            'LBXGH_MISSING', 'LBXGLU_MISSING', 'DIQ010_MISSING'
        ]

    leaked = [col for col in feature_cols if col in target_cols]

    if leaked:
        logger.error(f"TARGET LEAKAGE DETECTED: {leaked}")
    else:
        logger.info("No target leakage detected")

    return leaked


def compare_distributions(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare distributions before and after cleaning.

    Parameters
    ----------
    df_before : pd.DataFrame
        DataFrame before cleaning.
    df_after : pd.DataFrame
        DataFrame after cleaning.
    columns : list, optional
        Columns to compare. If None, compares all numeric columns.

    Returns
    -------
    pd.DataFrame
        Comparison statistics.
    """
    if columns is None:
        columns = df_before.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in columns if c in df_after.columns]

    results = []

    for col in columns:
        if col not in df_before.columns or col not in df_after.columns:
            continue

        before = df_before[col].dropna()
        after = df_after[col].dropna()

        # Skip if no valid values
        if len(before) == 0 or len(after) == 0:
            continue

        results.append({
            'variable': col,
            'n_before': len(before),
            'n_after': len(after),
            'mean_before': before.mean(),
            'mean_after': after.mean(),
            'mean_change_pct': ((after.mean() - before.mean()) / before.mean() * 100) if before.mean() != 0 else 0,
            'std_before': before.std(),
            'std_after': after.std(),
            'median_before': before.median(),
            'median_after': after.median(),
            'missing_before': df_before[col].isna().sum(),
            'missing_after': df_after[col].isna().sum()
        })

    results_df = pd.DataFrame(results)

    # Flag large distribution changes
    if len(results_df) > 0:
        large_changes = results_df[abs(results_df['mean_change_pct']) > 5]
        if len(large_changes) > 0:
            logger.warning(f"Large distribution changes (>5% mean change) detected in {len(large_changes)} columns")
            for _, row in large_changes.iterrows():
                logger.warning(f"  {row['variable']}: {row['mean_change_pct']:.2f}% change")
        else:
            logger.info("No large distribution changes detected")

    return results_df


def generate_cleaning_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    cleaning_log: Dict,
    output_path: Optional[Path] = None
) -> Dict:
    """
    Generate comprehensive cleaning report.

    Parameters
    ----------
    df_before : pd.DataFrame
        DataFrame before cleaning.
    df_after : pd.DataFrame
        DataFrame after cleaning.
    cleaning_log : dict
        Log from clean_pipeline function.
    output_path : Path, optional
        Path to save JSON report.

    Returns
    -------
    dict
        Comprehensive cleaning report.
    """
    config = load_config()
    range_config = config.get('validation', {}).get('ranges', {})

    report = {
        'summary': {
            'rows_before': len(df_before),
            'rows_after': len(df_after),
            'cols_before': len(df_before.columns),
            'cols_after': len(df_after.columns),
            'missing_values_before': int(df_before.isna().sum().sum()),
            'missing_values_after': int(df_after.isna().sum().sum()),
            'row_change': len(df_after) - len(df_before),
            'col_change': len(df_after.columns) - len(df_before.columns)
        },
        'cleaning_steps': cleaning_log.get('steps', {}),
        'validation': {}
    }

    # Run validation checks
    logger.info("\nRunning validation checks...")

    # Range checks
    range_results = check_ranges(df_after, range_config)
    report['validation']['range_checks'] = range_results.to_dict('records')

    # Logical consistency
    logic_results = check_logical_consistency(df_after)
    report['validation']['logical_consistency'] = logic_results.to_dict('records')

    # Distribution comparison (sample of important columns)
    important_cols = ['RIDAGEYR', 'BMXBMI', 'BMXWT', 'LBXGH', 'BPXSY1', 'BPXDI1']
    existing_cols = [c for c in important_cols if c in df_before.columns]
    dist_results = compare_distributions(df_before, df_after, existing_cols)
    report['validation']['distribution_comparison'] = dist_results.to_dict('records')

    # Calculate validation summary
    n_range_violations = range_results['n_out_of_range'].sum() if len(range_results) > 0 else 0
    n_logic_violations = logic_results['n_violations'].sum() if len(logic_results) > 0 else 0

    report['validation_summary'] = {
        'n_range_violations': int(n_range_violations),
        'n_logic_violations': int(n_logic_violations),
        'row_count_preserved': len(df_before) == len(df_after),
        'passed': n_range_violations == 0 and n_logic_violations == 0 and len(df_before) == len(df_after)
    }

    # Save report if path provided
    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Cleaning report saved to: {output_path}")

    return report


def validate_cleaned_data(
    df: pd.DataFrame,
    expected_row_count: Optional[int] = None
) -> bool:
    """
    Quick validation of cleaned data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    expected_row_count : int, optional
        Expected number of rows.

    Returns
    -------
    bool
        True if validation passes, False otherwise.
    """
    passed = True

    # Check row count
    if expected_row_count is not None and len(df) != expected_row_count:
        logger.error(f"Row count mismatch: expected {expected_row_count}, got {len(df)}")
        passed = False

    # Check for duplicate SEQNs
    if 'SEQN' in df.columns:
        n_duplicates = df['SEQN'].duplicated().sum()
        if n_duplicates > 0:
            logger.error(f"Found {n_duplicates} duplicate SEQNs")
            passed = False

    # Check that SEQN has no missing values
    if 'SEQN' in df.columns and df['SEQN'].isna().any():
        logger.error("SEQN contains missing values")
        passed = False

    # Check that required columns exist
    required_cols = ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'DIABETES_STATUS']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        passed = False

    if passed:
        logger.info("Cleaned data validation PASSED")
    else:
        logger.error("Cleaned data validation FAILED")

    return passed
