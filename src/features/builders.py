"""
NHANES Feature Engineering

This module creates derived features for the diabetes prediction project.
Each feature includes clinical/scientific rationale documented in the function.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conversion constants
LBS_TO_KG = 0.453592


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_config() -> dict:
    """Load the project configuration."""
    config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# DERIVED FEATURE FUNCTIONS
# Each function documents the clinical rationale and handles edge cases
# =============================================================================

def create_avg_blood_pressure(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create average blood pressure features from multiple readings.

    CLINICAL RATIONALE:
    - Blood pressure is measured 3-4 times in NHANES to account for white coat
      hypertension and measurement variability
    - Averaging multiple readings provides more stable, reliable BP estimates
    - Systolic BP ≥130 or Diastolic BP ≥80 indicates hypertension (2017 ACC/AHA)
    - Hypertension is a major diabetes comorbidity and cardiovascular risk factor

    CALCULATION:
    - AVG_SYS_BP = mean(BPXSY1, BPXSY2, BPXSY3), ignoring NaN
    - AVG_DIA_BP = mean(BPXDI1, BPXDI2, BPXDI3), ignoring NaN

    EDGE CASES:
    - If all readings are NaN, result is NaN
    - Diastolic BP = 0 is valid (can occur with aortic regurgitation)
    """
    df = df.copy()
    stats = {}

    # Systolic BP
    sys_cols = ['BPXSY1', 'BPXSY2', 'BPXSY3']
    available_sys = [c for c in sys_cols if c in df.columns]
    if available_sys:
        df['AVG_SYS_BP'] = df[available_sys].mean(axis=1, skipna=True)
        stats['AVG_SYS_BP'] = {
            'source_cols': available_sys,
            'n_valid': df['AVG_SYS_BP'].notna().sum(),
            'mean': float(df['AVG_SYS_BP'].mean()),
            'median': float(df['AVG_SYS_BP'].median()),
            'range': [float(df['AVG_SYS_BP'].min()), float(df['AVG_SYS_BP'].max())]
        }
        logger.info(f"Created AVG_SYS_BP: mean={stats['AVG_SYS_BP']['mean']:.1f}, "
                   f"n_valid={stats['AVG_SYS_BP']['n_valid']}")

    # Diastolic BP
    dia_cols = ['BPXDI1', 'BPXDI2', 'BPXDI3']
    available_dia = [c for c in dia_cols if c in df.columns]
    if available_dia:
        df['AVG_DIA_BP'] = df[available_dia].mean(axis=1, skipna=True)
        stats['AVG_DIA_BP'] = {
            'source_cols': available_dia,
            'n_valid': df['AVG_DIA_BP'].notna().sum(),
            'mean': float(df['AVG_DIA_BP'].mean()),
            'median': float(df['AVG_DIA_BP'].median()),
            'range': [float(df['AVG_DIA_BP'].min()), float(df['AVG_DIA_BP'].max())]
        }
        logger.info(f"Created AVG_DIA_BP: mean={stats['AVG_DIA_BP']['mean']:.1f}, "
                   f"n_valid={stats['AVG_DIA_BP']['n_valid']}")

    return df, stats


def create_total_water_intake(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create total water intake feature from dietary recall.

    CLINICAL RATIONALE:
    - Adequate hydration affects blood glucose regulation and kidney function
    - Low water intake is associated with higher risk of hyperglycemia
    - Combines plain water, tap water, and bottled water for total intake
    - Measured in grams (1g water ≈ 1mL)

    CALCULATION:
    - TOTAL_WATER = DR1_320Z + DR1_330Z + DR1BWATZ
    - Units: grams

    EDGE CASES:
    - If all components are NaN, result is NaN
    - If some components are NaN, sum available (treats NaN as 0)
    """
    df = df.copy()
    stats = {}

    water_cols = ['DR1_320Z', 'DR1_330Z', 'DR1BWATZ']
    available = [c for c in water_cols if c in df.columns]

    if available:
        # Sum available water columns, treating NaN as 0
        # But if ALL are NaN, result should be NaN
        df['TOTAL_WATER'] = df[available].sum(axis=1, skipna=True)
        all_nan_mask = df[available].isna().all(axis=1)
        df.loc[all_nan_mask, 'TOTAL_WATER'] = np.nan

        stats['TOTAL_WATER'] = {
            'source_cols': available,
            'n_valid': df['TOTAL_WATER'].notna().sum(),
            'mean': float(df['TOTAL_WATER'].mean()),
            'median': float(df['TOTAL_WATER'].median()),
            'range': [float(df['TOTAL_WATER'].min()), float(df['TOTAL_WATER'].max())]
        }
        logger.info(f"Created TOTAL_WATER: mean={stats['TOTAL_WATER']['mean']:.1f}g, "
                   f"n_valid={stats['TOTAL_WATER']['n_valid']}")

    return df, stats


def create_acr_ratio(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create Albumin-to-Creatinine Ratio (ACR) for kidney function assessment.

    CLINICAL RATIONALE:
    - ACR is a key marker of kidney damage (diabetic nephropathy)
    - Diabetes is the leading cause of chronic kidney disease
    - ACR ≥30 mg/g indicates microalbuminuria (early kidney damage)
    - ACR ≥300 mg/g indicates macroalbuminuria (overt kidney disease)
    - ACR is more reliable than albumin alone as it corrects for urine concentration

    CALCULATION:
    - ACR_RATIO = URXUMA (albumin, mg/L) / URXUCR (creatinine, mg/dL) * 100
    - Conversion: (mg/L) / (mg/dL * 10) = mg/g creatinine
    - Simplified: URXUMA / URXUCR gives approximate mg/g

    EDGE CASES:
    - If creatinine is 0 or very low, result is undefined (set to NaN)
    - Very high ACR (>3000) may indicate measurement error
    """
    df = df.copy()
    stats = {}

    if 'URXUMA' in df.columns and 'URXUCR' in df.columns:
        # Avoid division by zero or very small values
        valid_creatinine = df['URXUCR'] > 1  # mg/dL threshold

        df['ACR_RATIO'] = np.nan
        df.loc[valid_creatinine, 'ACR_RATIO'] = (
            df.loc[valid_creatinine, 'URXUMA'] / df.loc[valid_creatinine, 'URXUCR']
        )

        # Cap extreme values (likely errors)
        extreme_mask = df['ACR_RATIO'] > 10000
        n_extreme = extreme_mask.sum()
        if n_extreme > 0:
            logger.warning(f"Found {n_extreme} extreme ACR values (>10000), capping")
            df.loc[extreme_mask, 'ACR_RATIO'] = np.nan

        stats['ACR_RATIO'] = {
            'source_cols': ['URXUMA', 'URXUCR'],
            'n_valid': df['ACR_RATIO'].notna().sum(),
            'mean': float(df['ACR_RATIO'].mean()),
            'median': float(df['ACR_RATIO'].median()),
            'range': [float(df['ACR_RATIO'].min()), float(df['ACR_RATIO'].max())],
            'pct_abnormal_30': float((df['ACR_RATIO'] >= 30).mean() * 100),
            'pct_abnormal_300': float((df['ACR_RATIO'] >= 300).mean() * 100)
        }
        logger.info(f"Created ACR_RATIO: median={stats['ACR_RATIO']['median']:.1f}, "
                   f"abnormal (≥30): {stats['ACR_RATIO']['pct_abnormal_30']:.1f}%")

    return df, stats


def create_weight_change_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create weight change features from weight history.

    CLINICAL RATIONALE:
    - Weight gain is a major risk factor for Type 2 diabetes
    - Weight change trajectory matters: recent gain vs. stable obesity
    - Weight at age 25 represents pre-disease baseline for many
    - Weight loss from maximum may indicate intentional lifestyle change OR disease

    CALCULATIONS:
    - WEIGHT_CHANGE_10YR = BMXWT (kg) - WHD110 (lbs) * 0.453592
    - WEIGHT_CHANGE_25 = BMXWT (kg) - WHD120 (lbs) * 0.453592
    - WEIGHT_FROM_MAX = WHD140 (lbs) * 0.453592 - BMXWT (kg)

    Positive values indicate weight gain; negative indicate loss.
    WEIGHT_FROM_MAX is positive if current < max (weight lost from peak).

    EDGE CASES:
    - WHD110/120/140 are in pounds, BMXWT is in kg
    - Values of -7 (Refused) or -9 (Don't Know) should be treated as missing
    - Very large changes (>100 kg) may indicate data errors
    """
    df = df.copy()
    stats = {}

    # Special values to treat as missing
    special_values = {-7, -9, 7777, 9999}

    def clean_weight_lbs(series):
        """Clean weight in lbs, handling special values."""
        cleaned = series.copy()
        mask = cleaned.isin(special_values)
        cleaned.loc[mask] = np.nan
        return cleaned

    # Weight change from 10 years ago
    if 'BMXWT' in df.columns and 'WHD110' in df.columns:
        whd110_clean = clean_weight_lbs(df['WHD110'])
        whd110_kg = whd110_clean * LBS_TO_KG
        df['WEIGHT_CHANGE_10YR'] = df['BMXWT'] - whd110_kg

        stats['WEIGHT_CHANGE_10YR'] = {
            'source_cols': ['BMXWT', 'WHD110'],
            'n_valid': df['WEIGHT_CHANGE_10YR'].notna().sum(),
            'mean': float(df['WEIGHT_CHANGE_10YR'].mean()),
            'median': float(df['WEIGHT_CHANGE_10YR'].median()),
            'range': [float(df['WEIGHT_CHANGE_10YR'].min()), float(df['WEIGHT_CHANGE_10YR'].max())],
            'pct_gained': float((df['WEIGHT_CHANGE_10YR'] > 0).mean() * 100),
            'pct_lost': float((df['WEIGHT_CHANGE_10YR'] < 0).mean() * 100)
        }
        logger.info(f"Created WEIGHT_CHANGE_10YR: mean={stats['WEIGHT_CHANGE_10YR']['mean']:.1f}kg, "
                   f"gained: {stats['WEIGHT_CHANGE_10YR']['pct_gained']:.1f}%")

    # Weight change from age 25
    if 'BMXWT' in df.columns and 'WHD120' in df.columns:
        whd120_clean = clean_weight_lbs(df['WHD120'])
        whd120_kg = whd120_clean * LBS_TO_KG
        df['WEIGHT_CHANGE_25'] = df['BMXWT'] - whd120_kg

        stats['WEIGHT_CHANGE_25'] = {
            'source_cols': ['BMXWT', 'WHD120'],
            'n_valid': df['WEIGHT_CHANGE_25'].notna().sum(),
            'mean': float(df['WEIGHT_CHANGE_25'].mean()),
            'median': float(df['WEIGHT_CHANGE_25'].median()),
            'range': [float(df['WEIGHT_CHANGE_25'].min()), float(df['WEIGHT_CHANGE_25'].max())],
            'pct_gained': float((df['WEIGHT_CHANGE_25'] > 0).mean() * 100)
        }
        logger.info(f"Created WEIGHT_CHANGE_25: mean={stats['WEIGHT_CHANGE_25']['mean']:.1f}kg, "
                   f"gained: {stats['WEIGHT_CHANGE_25']['pct_gained']:.1f}%")

    # Weight from maximum (positive = lost weight)
    if 'BMXWT' in df.columns and 'WHD140' in df.columns:
        whd140_clean = clean_weight_lbs(df['WHD140'])
        whd140_kg = whd140_clean * LBS_TO_KG
        df['WEIGHT_FROM_MAX'] = whd140_kg - df['BMXWT']

        stats['WEIGHT_FROM_MAX'] = {
            'source_cols': ['BMXWT', 'WHD140'],
            'n_valid': df['WEIGHT_FROM_MAX'].notna().sum(),
            'mean': float(df['WEIGHT_FROM_MAX'].mean()),
            'median': float(df['WEIGHT_FROM_MAX'].median()),
            'range': [float(df['WEIGHT_FROM_MAX'].min()), float(df['WEIGHT_FROM_MAX'].max())],
            'pct_at_max': float((df['WEIGHT_FROM_MAX'] <= 0).mean() * 100)
        }
        logger.info(f"Created WEIGHT_FROM_MAX: mean={stats['WEIGHT_FROM_MAX']['mean']:.1f}kg, "
                   f"at/above max: {stats['WEIGHT_FROM_MAX']['pct_at_max']:.1f}%")

    return df, stats


def create_wake_time_diff(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create wake time difference between weekday and weekend.

    CLINICAL RATIONALE:
    - "Social jet lag" (different sleep schedules on weekdays vs weekends) is
      associated with metabolic dysfunction and diabetes risk
    - Irregular sleep patterns disrupt circadian rhythms affecting glucose metabolism
    - Large differences may indicate sleep debt accumulation during the week

    CALCULATION:
    - WAKE_TIME_DIFF = SLQ330 (weekend wake time) - SLQ310 (weekday wake time)
    - Units: hours
    - Positive = waking later on weekends

    EDGE CASES:
    - Times may be in "HH:MM" string format or HHMM numeric format
    - Need to handle both formats robustly
    - Special values (-7, -9) should be treated as missing
    """
    df = df.copy()
    stats = {}

    def time_to_hours(series):
        """Convert time to decimal hours. Handles both HH:MM string and HHMM numeric formats."""
        result = pd.Series(index=series.index, dtype=float)

        for idx, val in series.items():
            if pd.isna(val):
                result[idx] = np.nan
            elif isinstance(val, str):
                # Handle "HH:MM" string format
                try:
                    if ':' in val:
                        parts = val.split(':')
                        hours = int(parts[0]) + int(parts[1]) / 60
                        result[idx] = hours
                    else:
                        result[idx] = np.nan
                except (ValueError, IndexError):
                    result[idx] = np.nan
            elif isinstance(val, (int, float)):
                # Handle HHMM numeric format
                if val in {-7, -9, 7777, 9999}:
                    result[idx] = np.nan
                else:
                    val = int(val)
                    hours = (val // 100) + (val % 100) / 60
                    result[idx] = hours
            else:
                result[idx] = np.nan

        return result

    if 'SLQ310' in df.columns and 'SLQ330' in df.columns:
        weekday_hours = time_to_hours(df['SLQ310'])
        weekend_hours = time_to_hours(df['SLQ330'])

        df['WAKE_TIME_DIFF'] = weekend_hours - weekday_hours

        valid_diff = df['WAKE_TIME_DIFF'].dropna()
        if len(valid_diff) > 0:
            stats['WAKE_TIME_DIFF'] = {
                'source_cols': ['SLQ310', 'SLQ330'],
                'n_valid': int(df['WAKE_TIME_DIFF'].notna().sum()),
                'mean': float(valid_diff.mean()),
                'median': float(valid_diff.median()),
                'range': [float(valid_diff.min()), float(valid_diff.max())],
                'pct_later_weekend': float((valid_diff > 0).mean() * 100)
            }
            logger.info(f"Created WAKE_TIME_DIFF: mean={stats['WAKE_TIME_DIFF']['mean']:.2f}hrs, "
                       f"later on weekends: {stats['WAKE_TIME_DIFF']['pct_later_weekend']:.1f}%")
        else:
            stats['WAKE_TIME_DIFF'] = {'n_valid': 0}
            logger.warning("WAKE_TIME_DIFF: No valid values computed")

    return df, stats


def create_wake_time_hours(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Convert wake time columns from time strings to decimal hours.

    CLINICAL RATIONALE:
    - Wake time reflects circadian rhythm patterns
    - Early vs late chronotype associated with metabolic health differences
    - Late chronotype (waking later) associated with higher diabetes risk
    - Capturing actual wake time (not just difference) provides additional signal

    CALCULATION:
    - SLQ310_HOURS = weekday wake time in decimal hours (e.g., "05:30" -> 5.5)
    - SLQ330_HOURS = weekend wake time in decimal hours

    EDGE CASES:
    - Times stored as "HH:MM" string format
    - Special values (-7, -9) treated as missing
    - Times after midnight (e.g., 01:00) are valid for late sleepers
    """
    df = df.copy()
    stats = {}

    def time_to_decimal_hours(series):
        """Convert time to decimal hours. Handles both HH:MM string and HHMM numeric formats."""
        result = pd.Series(index=series.index, dtype=float)

        for idx, val in series.items():
            if pd.isna(val):
                result[idx] = np.nan
            elif isinstance(val, str):
                # Handle "HH:MM" string format
                try:
                    if ':' in val:
                        parts = val.split(':')
                        hours = int(parts[0]) + int(parts[1]) / 60
                        result[idx] = hours
                    else:
                        result[idx] = np.nan
                except (ValueError, IndexError):
                    result[idx] = np.nan
            elif isinstance(val, (int, float)):
                # Handle HHMM numeric format or special values
                if val in {-7, -9, 7777, 9999}:
                    result[idx] = np.nan
                else:
                    val = int(val)
                    hours = (val // 100) + (val % 100) / 60
                    result[idx] = hours
            else:
                result[idx] = np.nan

        return result

    # Convert weekday wake time
    if 'SLQ310' in df.columns:
        df['SLQ310_HOURS'] = time_to_decimal_hours(df['SLQ310'])

        valid = df['SLQ310_HOURS'].dropna()
        if len(valid) > 0:
            stats['SLQ310_HOURS'] = {
                'source_col': 'SLQ310',
                'n_valid': int(len(valid)),
                'mean': float(valid.mean()),
                'median': float(valid.median()),
                'range': [float(valid.min()), float(valid.max())],
                'pct_before_6am': float((valid < 6).mean() * 100),
                'pct_after_8am': float((valid > 8).mean() * 100)
            }
            logger.info(f"Created SLQ310_HOURS (weekday wake): mean={stats['SLQ310_HOURS']['mean']:.1f}h, "
                       f"before 6am: {stats['SLQ310_HOURS']['pct_before_6am']:.1f}%")

    # Convert weekend wake time
    if 'SLQ330' in df.columns:
        df['SLQ330_HOURS'] = time_to_decimal_hours(df['SLQ330'])

        valid = df['SLQ330_HOURS'].dropna()
        if len(valid) > 0:
            stats['SLQ330_HOURS'] = {
                'source_col': 'SLQ330',
                'n_valid': int(len(valid)),
                'mean': float(valid.mean()),
                'median': float(valid.median()),
                'range': [float(valid.min()), float(valid.max())],
                'pct_before_6am': float((valid < 6).mean() * 100),
                'pct_after_8am': float((valid > 8).mean() * 100)
            }
            logger.info(f"Created SLQ330_HOURS (weekend wake): mean={stats['SLQ330_HOURS']['mean']:.1f}h, "
                       f"after 8am: {stats['SLQ330_HOURS']['pct_after_8am']:.1f}%")

    return df, stats


def create_waist_height_ratio(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create waist-to-height ratio for central obesity assessment.

    CLINICAL RATIONALE:
    - Waist-to-height ratio is a better predictor of diabetes risk than BMI alone
    - Captures central/visceral adiposity which is more metabolically harmful
    - Simple rule: "keep your waist to less than half your height"
    - Ratio >0.5 indicates increased cardiometabolic risk
    - Ratio >0.6 indicates substantially increased risk

    CALCULATION:
    - WAIST_HEIGHT_RATIO = BMXWAIST (cm) / BMXHT (cm)

    EDGE CASES:
    - Height of 0 would cause division error (set to NaN)
    - Ratios >1.0 are possible in extreme obesity but rare
    """
    df = df.copy()
    stats = {}

    if 'BMXWAIST' in df.columns and 'BMXHT' in df.columns:
        valid_height = df['BMXHT'] > 0

        df['WAIST_HEIGHT_RATIO'] = np.nan
        df.loc[valid_height, 'WAIST_HEIGHT_RATIO'] = (
            df.loc[valid_height, 'BMXWAIST'] / df.loc[valid_height, 'BMXHT']
        )

        stats['WAIST_HEIGHT_RATIO'] = {
            'source_cols': ['BMXWAIST', 'BMXHT'],
            'n_valid': df['WAIST_HEIGHT_RATIO'].notna().sum(),
            'mean': float(df['WAIST_HEIGHT_RATIO'].mean()),
            'median': float(df['WAIST_HEIGHT_RATIO'].median()),
            'range': [float(df['WAIST_HEIGHT_RATIO'].min()), float(df['WAIST_HEIGHT_RATIO'].max())],
            'pct_elevated_0.5': float((df['WAIST_HEIGHT_RATIO'] > 0.5).mean() * 100),
            'pct_high_0.6': float((df['WAIST_HEIGHT_RATIO'] > 0.6).mean() * 100)
        }
        logger.info(f"Created WAIST_HEIGHT_RATIO: mean={stats['WAIST_HEIGHT_RATIO']['mean']:.3f}, "
                   f">0.5: {stats['WAIST_HEIGHT_RATIO']['pct_elevated_0.5']:.1f}%")

    return df, stats


def create_sat_fat_percentage(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create saturated fat percentage of total fat intake.

    CLINICAL RATIONALE:
    - Saturated fat intake is linked to insulin resistance and diabetes risk
    - AHA recommends <10% of calories from saturated fat
    - The ratio matters more than absolute amounts for dietary quality assessment
    - High saturated/total fat ratio indicates poor fat quality choices

    CALCULATION:
    - SAT_FAT_PCT = (DR1TSFAT / DR1TTFAT) * 100
    - Both in grams

    EDGE CASES:
    - Total fat of 0 would cause division error
    - Percentage >100 is impossible (error if occurs)
    """
    df = df.copy()
    stats = {}

    if 'DR1TSFAT' in df.columns and 'DR1TTFAT' in df.columns:
        valid_fat = df['DR1TTFAT'] > 0

        df['SAT_FAT_PCT'] = np.nan
        df.loc[valid_fat, 'SAT_FAT_PCT'] = (
            df.loc[valid_fat, 'DR1TSFAT'] / df.loc[valid_fat, 'DR1TTFAT'] * 100
        )

        # Sanity check: percentage should be 0-100
        invalid = (df['SAT_FAT_PCT'] < 0) | (df['SAT_FAT_PCT'] > 100)
        if invalid.sum() > 0:
            logger.warning(f"Found {invalid.sum()} invalid SAT_FAT_PCT values, setting to NaN")
            df.loc[invalid, 'SAT_FAT_PCT'] = np.nan

        stats['SAT_FAT_PCT'] = {
            'source_cols': ['DR1TSFAT', 'DR1TTFAT'],
            'n_valid': df['SAT_FAT_PCT'].notna().sum(),
            'mean': float(df['SAT_FAT_PCT'].mean()),
            'median': float(df['SAT_FAT_PCT'].median()),
            'range': [float(df['SAT_FAT_PCT'].min()), float(df['SAT_FAT_PCT'].max())]
        }
        logger.info(f"Created SAT_FAT_PCT: mean={stats['SAT_FAT_PCT']['mean']:.1f}%, "
                   f"n_valid={stats['SAT_FAT_PCT']['n_valid']}")

    return df, stats


def create_pulse_pressure(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create pulse pressure from average blood pressure readings.

    CLINICAL RATIONALE:
    - Pulse pressure = Systolic BP - Diastolic BP
    - Reflects arterial stiffness and large artery compliance
    - Normal: 40-60 mmHg
    - High pulse pressure (>60) is an independent cardiovascular risk factor
    - Particularly important in older adults and diabetics
    - Wide pulse pressure indicates stiff arteries, common in diabetes

    CALCULATION:
    - PULSE_PRESSURE = AVG_SYS_BP - AVG_DIA_BP

    EDGE CASES:
    - Requires AVG_SYS_BP and AVG_DIA_BP to be calculated first
    - Negative values are impossible (indicates data error)
    """
    df = df.copy()
    stats = {}

    if 'AVG_SYS_BP' in df.columns and 'AVG_DIA_BP' in df.columns:
        df['PULSE_PRESSURE'] = df['AVG_SYS_BP'] - df['AVG_DIA_BP']

        # Sanity check: pulse pressure should be positive
        invalid = df['PULSE_PRESSURE'] < 0
        if invalid.sum() > 0:
            logger.warning(f"Found {invalid.sum()} negative pulse pressure values, setting to NaN")
            df.loc[invalid, 'PULSE_PRESSURE'] = np.nan

        valid = df['PULSE_PRESSURE'].dropna()
        stats['PULSE_PRESSURE'] = {
            'source_cols': ['AVG_SYS_BP', 'AVG_DIA_BP'],
            'n_valid': int(len(valid)),
            'mean': float(valid.mean()),
            'median': float(valid.median()),
            'range': [float(valid.min()), float(valid.max())],
            'pct_elevated_60': float((valid > 60).mean() * 100)
        }
        logger.info(f"Created PULSE_PRESSURE: mean={stats['PULSE_PRESSURE']['mean']:.1f}mmHg, "
                   f">60mmHg: {stats['PULSE_PRESSURE']['pct_elevated_60']:.1f}%")

    return df, stats


def create_mean_arterial_pressure(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create Mean Arterial Pressure (MAP).

    CLINICAL RATIONALE:
    - MAP represents the average pressure during a cardiac cycle
    - MAP = (SYS + 2*DIA) / 3 (approximation, weights diastole more since heart spends more time in diastole)
    - Normal MAP: 70-100 mmHg
    - MAP < 60 indicates inadequate organ perfusion
    - MAP > 100 indicates hypertension
    - Single value more clinically meaningful than separate SYS/DIA for some applications

    CALCULATION:
    - MAP = (AVG_SYS_BP + 2 * AVG_DIA_BP) / 3
    """
    df = df.copy()
    stats = {}

    if 'AVG_SYS_BP' in df.columns and 'AVG_DIA_BP' in df.columns:
        df['MAP'] = (df['AVG_SYS_BP'] + 2 * df['AVG_DIA_BP']) / 3

        valid = df['MAP'].dropna()
        stats['MAP'] = {
            'source_cols': ['AVG_SYS_BP', 'AVG_DIA_BP'],
            'n_valid': int(len(valid)),
            'mean': float(valid.mean()),
            'median': float(valid.median()),
            'range': [float(valid.min()), float(valid.max())],
            'pct_hypertensive_100': float((valid > 100).mean() * 100)
        }
        logger.info(f"Created MAP: mean={stats['MAP']['mean']:.1f}mmHg, "
                   f">100mmHg: {stats['MAP']['pct_hypertensive_100']:.1f}%")

    return df, stats


def create_bp_variability(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create blood pressure variability (standard deviation of readings).

    CLINICAL RATIONALE:
    - High visit-to-visit or reading-to-reading BP variability is an independent
      cardiovascular risk factor, separate from mean BP level
    - High variability associated with target organ damage, stroke risk
    - May indicate autonomic dysfunction common in diabetes
    - Provides information beyond the average BP

    CALCULATION:
    - BP_VARIABILITY = std(BPXSY1, BPXSY2, BPXSY3)
    - Uses systolic readings as they have more prognostic value

    EDGE CASES:
    - If only 1 reading available, std is undefined (NaN)
    - Very high variability (>20 mmHg) may indicate measurement issues
    """
    df = df.copy()
    stats = {}

    sys_cols = ['BPXSY1', 'BPXSY2', 'BPXSY3']
    available = [c for c in sys_cols if c in df.columns]

    if len(available) >= 2:
        # Calculate std across readings (axis=1), requires at least 2 non-NaN values
        df['BP_VARIABILITY'] = df[available].std(axis=1, skipna=True, ddof=1)

        valid = df['BP_VARIABILITY'].dropna()
        stats['BP_VARIABILITY'] = {
            'source_cols': available,
            'n_valid': int(len(valid)),
            'mean': float(valid.mean()),
            'median': float(valid.median()),
            'range': [float(valid.min()), float(valid.max())],
            'pct_high_10': float((valid > 10).mean() * 100)
        }
        logger.info(f"Created BP_VARIABILITY: mean={stats['BP_VARIABILITY']['mean']:.1f}mmHg, "
                   f">10mmHg: {stats['BP_VARIABILITY']['pct_high_10']:.1f}%")

    return df, stats


def create_carb_fiber_ratio(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create carbohydrate-to-fiber ratio.

    CLINICAL RATIONALE:
    - Fiber slows carbohydrate absorption, reducing glycemic impact
    - Low ratio (more fiber per carb) = healthier, slower glucose release
    - High ratio = refined carbs, rapid glucose spikes
    - ADA recommends 14g fiber per 1000 kcal; typical American diet is much lower
    - Ratio >15:1 indicates poor carb quality

    CALCULATION:
    - CARB_FIBER_RATIO = DR1TCARB / DR1TFIBE
    - Both in grams

    EDGE CASES:
    - Fiber = 0 would cause division error
    - Very high ratios (>50) indicate very low fiber intake
    """
    df = df.copy()
    stats = {}

    if 'DR1TCARB' in df.columns and 'DR1TFIBE' in df.columns:
        valid_fiber = df['DR1TFIBE'] > 0

        df['CARB_FIBER_RATIO'] = np.nan
        df.loc[valid_fiber, 'CARB_FIBER_RATIO'] = (
            df.loc[valid_fiber, 'DR1TCARB'] / df.loc[valid_fiber, 'DR1TFIBE']
        )

        valid = df['CARB_FIBER_RATIO'].dropna()
        stats['CARB_FIBER_RATIO'] = {
            'source_cols': ['DR1TCARB', 'DR1TFIBE'],
            'n_valid': int(len(valid)),
            'mean': float(valid.mean()),
            'median': float(valid.median()),
            'range': [float(valid.min()), float(valid.max())],
            'pct_poor_15': float((valid > 15).mean() * 100)
        }
        logger.info(f"Created CARB_FIBER_RATIO: mean={stats['CARB_FIBER_RATIO']['mean']:.1f}, "
                   f">15 (poor): {stats['CARB_FIBER_RATIO']['pct_poor_15']:.1f}%")

    return df, stats


def create_sugar_carb_ratio(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create sugar-to-carbohydrate ratio.

    CLINICAL RATIONALE:
    - Indicates proportion of carbs from simple sugars vs complex carbs
    - High ratio = more refined/simple sugars, faster glucose absorption
    - WHO recommends <10% of calories from free sugars
    - Higher sugar ratio associated with insulin resistance and diabetes risk

    CALCULATION:
    - SUGAR_CARB_RATIO = (DR1TSUGR / DR1TCARB) * 100
    - Expressed as percentage

    EDGE CASES:
    - Total carbs = 0 would cause division error
    - Ratio should be 0-100% (sugars are subset of carbs)
    """
    df = df.copy()
    stats = {}

    if 'DR1TSUGR' in df.columns and 'DR1TCARB' in df.columns:
        valid_carb = df['DR1TCARB'] > 0

        df['SUGAR_CARB_RATIO'] = np.nan
        df.loc[valid_carb, 'SUGAR_CARB_RATIO'] = (
            df.loc[valid_carb, 'DR1TSUGR'] / df.loc[valid_carb, 'DR1TCARB'] * 100
        )

        # Sanity check
        invalid = (df['SUGAR_CARB_RATIO'] < 0) | (df['SUGAR_CARB_RATIO'] > 100)
        if invalid.sum() > 0:
            df.loc[invalid, 'SUGAR_CARB_RATIO'] = np.nan

        valid = df['SUGAR_CARB_RATIO'].dropna()
        stats['SUGAR_CARB_RATIO'] = {
            'source_cols': ['DR1TSUGR', 'DR1TCARB'],
            'n_valid': int(len(valid)),
            'mean': float(valid.mean()),
            'median': float(valid.median()),
            'range': [float(valid.min()), float(valid.max())]
        }
        logger.info(f"Created SUGAR_CARB_RATIO: mean={stats['SUGAR_CARB_RATIO']['mean']:.1f}%")

    return df, stats


def create_phq9_score(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create PHQ-9 depression screening total score.

    CLINICAL RATIONALE:
    - PHQ-9 is a validated depression screening tool
    - Depression is bidirectionally linked to diabetes:
      * Depression increases diabetes risk 60%
      * Diabetes increases depression risk 2-3x
    - Depression affects self-care, medication adherence, glycemic control
    - Each item scored 0-3, total 0-27

    SEVERITY CATEGORIES:
    - 0-4: Minimal depression
    - 5-9: Mild depression
    - 10-14: Moderate depression
    - 15-19: Moderately severe
    - 20-27: Severe depression

    CALCULATION:
    - PHQ9_SCORE = sum(DPQ010, DPQ020, ..., DPQ090)
    - Only calculated if at least 7 of 9 items answered

    EDGE CASES:
    - Special values (-7 Refused, -9 Don't know) treated as missing
    - Partial responses: require ≥7 items for valid score
    """
    df = df.copy()
    stats = {}

    phq_cols = ['DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050',
                'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090']
    available = [c for c in phq_cols if c in df.columns]

    if len(available) >= 7:
        # Create working copy with special values as NaN
        phq_data = df[available].copy()
        for col in available:
            phq_data.loc[phq_data[col].isin({-7, -9, 7, 9, 77, 99}), col] = np.nan

        # Count valid responses per row
        valid_count = phq_data.notna().sum(axis=1)

        # Calculate sum only if ≥7 items answered
        df['PHQ9_SCORE'] = np.nan
        sufficient = valid_count >= 7
        df.loc[sufficient, 'PHQ9_SCORE'] = phq_data.loc[sufficient].sum(axis=1, skipna=True)

        valid = df['PHQ9_SCORE'].dropna()
        if len(valid) > 0:
            stats['PHQ9_SCORE'] = {
                'source_cols': available,
                'n_valid': int(len(valid)),
                'mean': float(valid.mean()),
                'median': float(valid.median()),
                'range': [float(valid.min()), float(valid.max())],
                'pct_moderate_10': float((valid >= 10).mean() * 100),
                'pct_severe_15': float((valid >= 15).mean() * 100)
            }
            logger.info(f"Created PHQ9_SCORE: mean={stats['PHQ9_SCORE']['mean']:.1f}, "
                       f"moderate+ (≥10): {stats['PHQ9_SCORE']['pct_moderate_10']:.1f}%")

    return df, stats


def create_tg_hdl_ratio(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create Triglyceride-to-HDL ratio.

    CLINICAL RATIONALE:
    - Excellent surrogate marker for insulin resistance
    - Better predictor of cardiovascular risk than LDL in many studies
    - Ratio >3.0 strongly associated with insulin resistance
    - Ratio >5.0 indicates high atherogenic risk
    - Particularly useful because it's calculated from standard lipid panel

    CALCULATION:
    - TG_HDL_RATIO = LBXTR / LBDHDD
    - Both in mg/dL

    EDGE CASES:
    - HDL = 0 would cause division error (very rare)
    - Very high ratios (>10) indicate severe dyslipidemia
    """
    df = df.copy()
    stats = {}

    if 'LBXTR' in df.columns and 'LBDHDD' in df.columns:
        valid_hdl = df['LBDHDD'] > 0

        df['TG_HDL_RATIO'] = np.nan
        df.loc[valid_hdl, 'TG_HDL_RATIO'] = (
            df.loc[valid_hdl, 'LBXTR'] / df.loc[valid_hdl, 'LBDHDD']
        )

        valid = df['TG_HDL_RATIO'].dropna()
        if len(valid) > 0:
            stats['TG_HDL_RATIO'] = {
                'source_cols': ['LBXTR', 'LBDHDD'],
                'n_valid': int(len(valid)),
                'mean': float(valid.mean()),
                'median': float(valid.median()),
                'range': [float(valid.min()), float(valid.max())],
                'pct_insulin_resistant_3': float((valid > 3.0).mean() * 100),
                'pct_high_risk_5': float((valid > 5.0).mean() * 100)
            }
            logger.info(f"Created TG_HDL_RATIO: mean={stats['TG_HDL_RATIO']['mean']:.1f}, "
                       f">3.0 (IR): {stats['TG_HDL_RATIO']['pct_insulin_resistant_3']:.1f}%")

    return df, stats


def create_non_hdl_cholesterol(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create Non-HDL cholesterol.

    CLINICAL RATIONALE:
    - Non-HDL = Total cholesterol - HDL
    - Captures all atherogenic lipoproteins (LDL + VLDL + IDL + Lp(a))
    - Better predictor of cardiovascular risk than LDL alone
    - Doesn't require fasting (unlike LDL calculation)
    - AHA/ACC guidelines use non-HDL as secondary target after LDL

    THRESHOLDS:
    - Optimal: <130 mg/dL
    - Near optimal: 130-159 mg/dL
    - Borderline high: 160-189 mg/dL
    - High: 190-219 mg/dL
    - Very high: ≥220 mg/dL

    CALCULATION:
    - NON_HDL_CHOL = LBXTC - LBDHDD
    """
    df = df.copy()
    stats = {}

    if 'LBXTC' in df.columns and 'LBDHDD' in df.columns:
        df['NON_HDL_CHOL'] = df['LBXTC'] - df['LBDHDD']

        # Sanity check: should be positive
        invalid = df['NON_HDL_CHOL'] < 0
        if invalid.sum() > 0:
            df.loc[invalid, 'NON_HDL_CHOL'] = np.nan

        valid = df['NON_HDL_CHOL'].dropna()
        if len(valid) > 0:
            stats['NON_HDL_CHOL'] = {
                'source_cols': ['LBXTC', 'LBDHDD'],
                'n_valid': int(len(valid)),
                'mean': float(valid.mean()),
                'median': float(valid.median()),
                'range': [float(valid.min()), float(valid.max())],
                'pct_high_160': float((valid >= 160).mean() * 100)
            }
            logger.info(f"Created NON_HDL_CHOL: mean={stats['NON_HDL_CHOL']['mean']:.1f}mg/dL, "
                       f"≥160 (high): {stats['NON_HDL_CHOL']['pct_high_160']:.1f}%")

    return df, stats


def create_cvd_history_flag(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create binary flag for any cardiovascular disease history.

    CLINICAL RATIONALE:
    - CVD and diabetes share risk factors and pathophysiology
    - Having CVD indicates metabolic dysfunction and vascular damage
    - Diabetes patients with CVD have much higher mortality
    - Aggregating multiple CVD conditions into single flag improves signal

    CONDITIONS INCLUDED:
    - MCQ160B: Congestive heart failure
    - MCQ160C: Coronary heart disease
    - MCQ160D: Angina/angina pectoris
    - MCQ160E: Heart attack (myocardial infarction)
    - MCQ160F: Stroke

    CALCULATION:
    - ANY_CVD = 1 if any condition = 1 (Yes), else 0
    """
    df = df.copy()
    stats = {}

    cvd_cols = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    available = [c for c in cvd_cols if c in df.columns]

    if available:
        # 1 = Yes for these questions
        cvd_data = df[available].copy()

        # Any CVD = 1 if any condition is 1
        df['ANY_CVD'] = (cvd_data == 1).any(axis=1).astype(float)

        # If all CVD columns are NaN, result should be NaN
        all_nan = cvd_data.isna().all(axis=1)
        df.loc[all_nan, 'ANY_CVD'] = np.nan

        valid = df['ANY_CVD'].dropna()
        if len(valid) > 0:
            stats['ANY_CVD'] = {
                'source_cols': available,
                'n_valid': int(len(valid)),
                'pct_with_cvd': float(valid.mean() * 100),
                'n_with_cvd': int(valid.sum())
            }
            logger.info(f"Created ANY_CVD: {stats['ANY_CVD']['pct_with_cvd']:.1f}% have CVD history")

    return df, stats


def create_sleep_duration_diff(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create sleep duration difference between weekend and weekday.

    CLINICAL RATIONALE:
    - Complements WAKE_TIME_DIFF (social jet lag)
    - Large differences indicate sleep debt during the week
    - Sleeping longer on weekends to "catch up" suggests chronic sleep deprivation
    - Sleep deprivation impairs glucose tolerance and insulin sensitivity

    CALCULATION:
    - SLEEP_DURATION_DIFF = SLD013 (weekend hours) - SLD012 (weekday hours)
    - Positive = sleeping more on weekends

    EDGE CASES:
    - Special values (-7, -9) treated as missing
    - Very large differences (>6 hours) may indicate data issues
    """
    df = df.copy()
    stats = {}

    if 'SLD012' in df.columns and 'SLD013' in df.columns:
        weekday = df['SLD012'].copy()
        weekend = df['SLD013'].copy()

        # Handle special values
        weekday.loc[weekday.isin({-7, -9, 77, 99})] = np.nan
        weekend.loc[weekend.isin({-7, -9, 77, 99})] = np.nan

        df['SLEEP_DURATION_DIFF'] = weekend - weekday

        valid = df['SLEEP_DURATION_DIFF'].dropna()
        if len(valid) > 0:
            stats['SLEEP_DURATION_DIFF'] = {
                'source_cols': ['SLD012', 'SLD013'],
                'n_valid': int(len(valid)),
                'mean': float(valid.mean()),
                'median': float(valid.median()),
                'range': [float(valid.min()), float(valid.max())],
                'pct_more_weekend': float((valid > 0).mean() * 100)
            }
            logger.info(f"Created SLEEP_DURATION_DIFF: mean={stats['SLEEP_DURATION_DIFF']['mean']:.2f}hrs, "
                       f"more on weekends: {stats['SLEEP_DURATION_DIFF']['pct_more_weekend']:.1f}%")

    return df, stats


# =============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# =============================================================================

def create_all_derived_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create all derived features defined in the PRD.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw features.

    Returns
    -------
    tuple
        (df_with_features, feature_stats) documenting all created features.
    """
    logger.info("=" * 60)
    logger.info("Creating derived features")
    logger.info("=" * 60)

    all_stats = {}

    # Blood pressure averages
    df, stats = create_avg_blood_pressure(df)
    all_stats.update(stats)

    # Total water intake
    df, stats = create_total_water_intake(df)
    all_stats.update(stats)

    # ACR ratio (kidney function)
    df, stats = create_acr_ratio(df)
    all_stats.update(stats)

    # Weight change features
    df, stats = create_weight_change_features(df)
    all_stats.update(stats)

    # Wake time difference
    df, stats = create_wake_time_diff(df)
    all_stats.update(stats)

    # Wake time in decimal hours (converts time strings to numeric)
    df, stats = create_wake_time_hours(df)
    all_stats.update(stats)

    # Waist-to-height ratio
    df, stats = create_waist_height_ratio(df)
    all_stats.update(stats)

    # Saturated fat percentage
    df, stats = create_sat_fat_percentage(df)
    all_stats.update(stats)

    # --- NEW FEATURES (Phase 4 additions) ---

    # Pulse pressure (arterial stiffness)
    df, stats = create_pulse_pressure(df)
    all_stats.update(stats)

    # Mean arterial pressure
    df, stats = create_mean_arterial_pressure(df)
    all_stats.update(stats)

    # BP variability
    df, stats = create_bp_variability(df)
    all_stats.update(stats)

    # Carb-to-fiber ratio
    df, stats = create_carb_fiber_ratio(df)
    all_stats.update(stats)

    # Sugar-to-carb ratio
    df, stats = create_sugar_carb_ratio(df)
    all_stats.update(stats)

    # PHQ-9 depression score
    df, stats = create_phq9_score(df)
    all_stats.update(stats)

    # Triglyceride-to-HDL ratio (insulin resistance marker)
    df, stats = create_tg_hdl_ratio(df)
    all_stats.update(stats)

    # Non-HDL cholesterol
    df, stats = create_non_hdl_cholesterol(df)
    all_stats.update(stats)

    # Any CVD history flag
    df, stats = create_cvd_history_flag(df)
    all_stats.update(stats)

    # Sleep duration difference
    df, stats = create_sleep_duration_diff(df)
    all_stats.update(stats)

    logger.info("=" * 60)
    logger.info(f"Created {len(all_stats)} derived features")
    logger.info("=" * 60)

    return df, all_stats


# =============================================================================
# FEATURE SET DEFINITIONS
# =============================================================================

def get_feature_sets() -> Dict[str, Dict]:
    """
    Define the feature sets for modeling.

    Returns two sets:
    1. with_labs: All features including laboratory values
    2. without_labs: Excludes lab values (simulates screening without blood tests)

    Returns
    -------
    dict
        Feature set definitions with column lists.
    """

    # Laboratory features (require blood draw)
    lab_features = [
        # Lipid panel
        'LBXTC',      # Total cholesterol
        'LBDHDD',     # HDL cholesterol
        'LBDLDL',     # LDL cholesterol
        'LBXTR',      # Triglycerides
        'TG_HDL_RATIO',   # Derived: insulin resistance marker
        'NON_HDL_CHOL',   # Derived: all atherogenic lipoproteins
        # Kidney function
        'URXUMA',     # Urine albumin
        'URXUCR',     # Urine creatinine
        'LBXSCR',     # Serum creatinine
        'ACR_RATIO',  # Derived: albumin-creatinine ratio
        # Liver function
        'LBXSATSI',   # ALT
        'LBXSASSI',   # AST
        'LBXSGTSI',   # GGT
        # Blood count
        'LBXWBCSI',   # WBC
        'LBXHCT',     # Hematocrit
        'LBXHGB',     # Hemoglobin
        'LBXPLTSI',   # Platelets
    ]

    # Demographic features
    demographic_features = [
        'RIDAGEYR',   # Age
        'RIAGENDR',   # Gender
    ]

    # Anthropometric features
    anthropometric_features = [
        'BMXBMI',     # BMI
        'BMXWT',      # Weight
        'BMXHT',      # Height
        'BMXWAIST',   # Waist circumference
        'WAIST_HEIGHT_RATIO',  # Derived
    ]

    # Weight history features
    weight_history_features = [
        'WHD110',     # Weight 10 yrs ago
        'WHD120',     # Weight at age 25
        'WHD140',     # Greatest weight
        'WHD130',     # Age at heaviest
        'WEIGHT_CHANGE_10YR',  # Derived
        'WEIGHT_CHANGE_25',    # Derived
        'WEIGHT_FROM_MAX',     # Derived
    ]

    # Blood pressure exam features
    bp_exam_features = [
        'BPXSY1', 'BPXSY2', 'BPXSY3',  # Systolic readings
        'BPXDI1', 'BPXDI2', 'BPXDI3',  # Diastolic readings
        'AVG_SYS_BP',     # Derived average systolic
        'AVG_DIA_BP',     # Derived average diastolic
        'PULSE_PRESSURE', # Derived: arterial stiffness
        'MAP',            # Derived: mean arterial pressure
        'BP_VARIABILITY', # Derived: reading-to-reading variability
    ]

    # Blood pressure/cholesterol questionnaire
    bp_questionnaire_features = [
        'BPQ020',     # Ever told high BP
        'BPQ040A',    # Taking Rx for hypertension
        'BPQ080',     # Told high cholesterol
        'BPQ090D',    # Told take Rx for cholesterol
        'BPQ100D',    # Now taking Rx
    ]

    # Dietary nutrient features
    dietary_nutrient_features = [
        'DR1TKCAL',   # Energy (kcal)
        'DR1TPROT',   # Protein (g)
        'DR1TCARB',   # Carbohydrates (g)
        'DR1TSUGR',   # Sugars (g)
        'DR1TFIBE',   # Fiber (g)
        'DR1TTFAT',   # Total fat (g)
        'DR1TSFAT',   # Saturated fat (g)
        'DR1TMFAT',   # Monounsaturated fat (g)
        'DR1TPFAT',   # Polyunsaturated fat (g)
        'DR1TSODI',   # Sodium (mg)
        'DR1TCAFF',   # Caffeine (mg)
        'DR1TALCO',   # Alcohol (g)
        'DR1_320Z',   # Plain water (g)
        'DR1_330Z',   # Tap water (g)
        'DR1BWATZ',   # Bottled water (g)
        'TOTAL_WATER',      # Derived total water
        'SAT_FAT_PCT',      # Derived sat fat %
        'CARB_FIBER_RATIO', # Derived: carb quality
        'SUGAR_CARB_RATIO', # Derived: simple vs complex carbs
    ]

    # Dietary behavior features
    dietary_behavior_features = [
        'DBQ700',     # How healthy is diet
        'DBD895',     # Meals not home prepared
        'DBD900',     # Meals from fast food/pizza
        'DBQ197',     # Milk consumption past 30 days
    ]

    # Lifestyle - Alcohol
    alcohol_features = [
        'ALQ130',     # Drinks/day
        'ALQ121',     # Frequency
    ]

    # Lifestyle - Smoking
    smoking_features = [
        'SMQ020',     # Ever smoked 100 cigs
        'SMQ040',     # Current status
        'SMD650',     # Cigarettes/day
    ]

    # Lifestyle - Physical Activity
    physical_activity_features = [
        'PAQ605',     # Vigorous work
        'PAQ620',     # Moderate work
        'PAQ635',     # Walk/bicycle
        'PAQ650',     # Vigorous recreational
        'PAQ665',     # Moderate recreational
        'PAD680',     # Sedentary min/day
    ]

    # Lifestyle - Sleep
    sleep_features = [
        'SLD012',     # Weekday hours
        'SLD013',     # Weekend hours
        'SLQ050',     # Sleep disorder
        'SLQ310_HOURS',   # Weekday wake time (decimal hours, derived from SLQ310)
        'SLQ330_HOURS',   # Weekend wake time (decimal hours, derived from SLQ330)
        'WAKE_TIME_DIFF',       # Derived: social jet lag (wake time)
        'SLEEP_DURATION_DIFF',  # Derived: sleep debt pattern
    ]

    # Cardiovascular
    cardiovascular_features = [
        'CDQ010',     # Shortness of breath on stairs
    ]

    # Medical history features
    medical_history_features = [
        'MCQ160B',    # CHF
        'MCQ160C',    # CHD
        'MCQ160D',    # Angina
        'MCQ160E',    # Heart attack
        'MCQ160F',    # Stroke
        'MCQ300C',    # Family history diabetes
        'MCQ160L',    # Liver condition
        'KIQ022',     # Weak/failing kidneys
        'MCQ220',     # Cancer/malignancy ever
        'ANY_CVD',    # Derived: any cardiovascular history
    ]

    # Mental health - PHQ-9 Depression
    depression_features = [
        'DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050',
        'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090',
        'PHQ9_SCORE',  # Derived: total depression score
    ]

    # Combine all non-lab features
    without_labs = (
        demographic_features +
        anthropometric_features +
        weight_history_features +
        bp_exam_features +
        bp_questionnaire_features +
        dietary_nutrient_features +
        dietary_behavior_features +
        alcohol_features +
        smoking_features +
        physical_activity_features +
        sleep_features +
        cardiovascular_features +
        medical_history_features +
        depression_features
    )

    # All features including labs
    with_labs = without_labs + lab_features

    return {
        'with_labs': {
            'features': with_labs,
            'description': 'All features including laboratory values (requires blood draw)',
            'n_features': len(with_labs),
            'categories': {
                'demographic': len(demographic_features),
                'anthropometric': len(anthropometric_features),
                'weight_history': len(weight_history_features),
                'bp_exam': len(bp_exam_features),
                'bp_questionnaire': len(bp_questionnaire_features),
                'dietary_nutrient': len(dietary_nutrient_features),
                'dietary_behavior': len(dietary_behavior_features),
                'alcohol': len(alcohol_features),
                'smoking': len(smoking_features),
                'physical_activity': len(physical_activity_features),
                'sleep': len(sleep_features),
                'cardiovascular': len(cardiovascular_features),
                'medical_history': len(medical_history_features),
                'depression': len(depression_features),
                'laboratory': len(lab_features),
            }
        },
        'without_labs': {
            'features': without_labs,
            'description': 'Features excluding laboratory values (no blood draw required)',
            'n_features': len(without_labs),
            'categories': {
                'demographic': len(demographic_features),
                'anthropometric': len(anthropometric_features),
                'weight_history': len(weight_history_features),
                'bp_exam': len(bp_exam_features),
                'bp_questionnaire': len(bp_questionnaire_features),
                'dietary_nutrient': len(dietary_nutrient_features),
                'dietary_behavior': len(dietary_behavior_features),
                'alcohol': len(alcohol_features),
                'smoking': len(smoking_features),
                'physical_activity': len(physical_activity_features),
                'sleep': len(sleep_features),
                'cardiovascular': len(cardiovascular_features),
                'medical_history': len(medical_history_features),
                'depression': len(depression_features),
            }
        },
        'lab_only': {
            'features': lab_features,
            'description': 'Laboratory features only',
            'n_features': len(lab_features),
        }
    }


def validate_feature_availability(
    df: pd.DataFrame,
    feature_set: List[str]
) -> Tuple[List[str], List[str], Dict]:
    """
    Check which features from a feature set are available in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_set : list
        List of feature names to check.

    Returns
    -------
    tuple
        (available_features, missing_features, availability_stats)
    """
    available = [f for f in feature_set if f in df.columns]
    missing = [f for f in feature_set if f not in df.columns]

    stats = {
        'total_requested': len(feature_set),
        'available': len(available),
        'missing': len(missing),
        'availability_pct': len(available) / len(feature_set) * 100 if feature_set else 0
    }

    return available, missing, stats


def prepare_modeling_data(
    df: pd.DataFrame,
    feature_set: str = 'with_labs',
    imputation: str = 'minimal',
    target_col: str = 'DIABETES_STATUS',
    max_missing_rate: float = 0.50,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Prepare final modeling dataset with specified feature set and imputation level.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with all features.
    feature_set : str
        Either 'with_labs' or 'without_labs'.
    imputation : str
        'minimal' - preserve NaN values (for tree models like LightGBM)
        'full' - impute all values and remove features >max_missing_rate (for linear models)
    target_col : str
        Name of target column.
    max_missing_rate : float
        For 'full' imputation, remove features with missing rate > this threshold.

    Returns
    -------
    tuple
        (X, y, metadata)
    """
    feature_sets = get_feature_sets()

    if feature_set not in feature_sets:
        raise ValueError(f"Unknown feature set: {feature_set}. Choose 'with_labs' or 'without_labs'")

    if imputation not in ['minimal', 'full']:
        raise ValueError(f"Unknown imputation: {imputation}. Choose 'minimal' or 'full'")

    # Get requested features
    requested_features = feature_sets[feature_set]['features']

    # Check availability
    available, missing, avail_stats = validate_feature_availability(df, requested_features)

    if missing:
        logger.warning(f"Missing {len(missing)} features: {missing}")

    # Extract X and y
    X = df[available].copy()
    y = df[target_col].copy() if target_col in df.columns else None

    # Remove rows with missing target
    if y is not None:
        valid_target = y.notna()
        X = X.loc[valid_target]
        y = y.loc[valid_target]

    # Track removed features
    removed_high_missing = []

    if imputation == 'full':
        # Remove features with >max_missing_rate missing
        missing_rates = X.isna().mean()
        high_missing = missing_rates[missing_rates > max_missing_rate].index.tolist()
        removed_high_missing = high_missing

        if high_missing:
            logger.info(f"Removing {len(high_missing)} features with >{max_missing_rate*100:.0f}% missing: {high_missing}")
            X = X.drop(columns=high_missing)

        # Impute remaining missing values
        # Use median for numeric columns
        for col in X.columns:
            if X[col].isna().any():
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                else:
                    # For other types, use mode
                    mode_val = X[col].mode()
                    if len(mode_val) > 0:
                        X[col] = X[col].fillna(mode_val[0])

        # Verify no NaN remaining
        remaining_nan = X.isna().sum().sum()
        if remaining_nan > 0:
            logger.warning(f"Warning: {remaining_nan} NaN values remain after imputation")

    metadata = {
        'feature_set': feature_set,
        'imputation': imputation,
        'n_features': X.shape[1],
        'n_samples': len(X),
        'missing_features_not_in_data': missing,
        'removed_high_missing': removed_high_missing,
        'availability': avail_stats,
        'has_nan': X.isna().any().any(),
    }

    logger.info(f"Prepared {feature_set}_{imputation} dataset: {X.shape[0]} samples, {X.shape[1]} features")

    return X, y, metadata


def create_all_modeling_datasets(
    df: pd.DataFrame,
    target_col: str = 'DIABETES_STATUS',
    output_dir: Optional[Path] = None,
    max_missing_rate: float = 0.50,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series, Dict]]:
    """
    Create all 4 modeling datasets:
    1. with_labs_minimal - for tree models (LightGBM)
    2. with_labs_full - for linear models (LogReg, MLP)
    3. without_labs_minimal - for tree models without lab data
    4. without_labs_full - for linear models without lab data

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with all features.
    target_col : str
        Name of target column.
    output_dir : Path, optional
        Directory to save datasets. If None, datasets are not saved.
    max_missing_rate : float
        For 'full' imputation, remove features with missing rate > this threshold.

    Returns
    -------
    dict
        Dictionary mapping dataset name to (X, y, metadata) tuple.
    """
    datasets = {}

    # Define all 4 combinations
    combinations = [
        ('with_labs', 'minimal'),
        ('with_labs', 'full'),
        ('without_labs', 'minimal'),
        ('without_labs', 'full'),
    ]

    for feature_set, imputation in combinations:
        dataset_name = f"{feature_set}_{imputation}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating dataset: {dataset_name}")
        logger.info(f"{'='*60}")

        X, y, metadata = prepare_modeling_data(
            df=df,
            feature_set=feature_set,
            imputation=imputation,
            target_col=target_col,
            max_missing_rate=max_missing_rate,
        )

        datasets[dataset_name] = (X, y, metadata)

        # Log summary
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Has NaN: {metadata['has_nan']}")
        if metadata['removed_high_missing']:
            logger.info(f"  Removed {len(metadata['removed_high_missing'])} high-missing features")

    # Save if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, (X, y, metadata) in datasets.items():
            X.to_parquet(output_dir / f"X_{name}.parquet")
            y.to_frame().to_parquet(output_dir / f"y_{name}.parquet")

            # Save metadata
            import json
            with open(output_dir / f"metadata_{name}.json", 'w') as f:
                # Convert any non-serializable items
                meta_save = {k: v for k, v in metadata.items()}
                json.dump(meta_save, f, indent=2, default=str)

        logger.info(f"\nSaved all datasets to {output_dir}")

    return datasets
