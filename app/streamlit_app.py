"""
NHANES Diabetes Risk Prediction App

A Streamlit application for predicting diabetes risk using LightGBM models
trained on NHANES 2015-2018 data.

Author: Tim Hall
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "advanced"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_models():
    """Load the trained LightGBM models."""
    models = {}

    # Classification models
    cls_with_labs = MODELS_DIR / "classification" / "lgb_with_labs.joblib"
    cls_without_labs = MODELS_DIR / "classification" / "lgb_without_labs.joblib"

    if cls_with_labs.exists():
        models['classification_with_labs'] = joblib.load(cls_with_labs)
    if cls_without_labs.exists():
        models['classification_without_labs'] = joblib.load(cls_without_labs)

    # Regression models
    reg_with_labs = MODELS_DIR / "regression" / "lgb_with_labs.joblib"
    reg_without_labs = MODELS_DIR / "regression" / "lgb_without_labs.joblib"

    if reg_with_labs.exists():
        models['regression_with_labs'] = joblib.load(reg_with_labs)
    if reg_without_labs.exists():
        models['regression_without_labs'] = joblib.load(reg_without_labs)

    return models


@st.cache_data
def load_feature_info():
    """Load feature metadata."""
    report_path = DATA_DIR / "feature_engineering_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_feature_order():
    """Load feature order from the processed data files."""
    feature_order = {}

    # Load from parquet files to get exact feature order used during training
    with_labs_path = DATA_DIR / "X_with_labs_minimal.parquet"
    without_labs_path = DATA_DIR / "X_without_labs_minimal.parquet"

    if with_labs_path.exists():
        df = pd.read_parquet(with_labs_path)
        feature_order['with_labs'] = list(df.columns)

    if without_labs_path.exists():
        df = pd.read_parquet(without_labs_path)
        feature_order['without_labs'] = list(df.columns)

    return feature_order


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Features required for each model type
FEATURES_WITH_LABS = [
    # Demographics
    'RIDAGEYR', 'RIAGENDR',
    # Anthropometric
    'BMXBMI', 'BMXWT', 'BMXHT', 'BMXWAIST', 'WAIST_HEIGHT_RATIO',
    # Weight history
    'WHD110', 'WHD120', 'WHD140', 'WHD130', 'WEIGHT_CHANGE_10YR', 'WEIGHT_CHANGE_25', 'WEIGHT_FROM_MAX',
    # Blood pressure exam
    'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXDI1', 'BPXDI2', 'BPXDI3',
    'AVG_SYS_BP', 'AVG_DIA_BP', 'PULSE_PRESSURE', 'MAP', 'BP_VARIABILITY',
    # Blood pressure/cholesterol questionnaire
    'BPQ020', 'BPQ040A', 'BPQ080', 'BPQ090D', 'BPQ100D',
    # Dietary nutrients
    'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 'DR1TFIBE', 'DR1TTFAT',
    'DR1TSFAT', 'DR1TMFAT', 'DR1TPFAT', 'DR1TSODI', 'DR1TCAFF', 'DR1TALCO',
    'DR1_320Z', 'DR1_330Z', 'DR1BWATZ', 'TOTAL_WATER', 'SAT_FAT_PCT',
    'CARB_FIBER_RATIO', 'SUGAR_CARB_RATIO',
    # Dietary behavior
    'DBQ700', 'DBD895', 'DBD900', 'DBQ197',
    # Alcohol
    'ALQ130', 'ALQ121',
    # Smoking
    'SMQ020', 'SMQ040', 'SMD650',
    # Physical activity
    'PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665', 'PAD680',
    # Sleep
    'SLD012', 'SLD013', 'SLQ050', 'SLQ310_HOURS', 'SLQ330_HOURS',
    'WAKE_TIME_DIFF', 'SLEEP_DURATION_DIFF',
    # Cardiovascular
    'CDQ010',
    # Medical history
    'MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F', 'MCQ300C',
    'MCQ160L', 'KIQ022', 'MCQ220', 'ANY_CVD',
    # Depression
    'DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070',
    'DPQ080', 'DPQ090', 'PHQ9_SCORE',
    # Laboratory
    'LBXTC', 'LBDHDD', 'LBDLDL', 'LBXTR', 'TG_HDL_RATIO', 'NON_HDL_CHOL',
    'URXUMA', 'URXUCR', 'LBXSCR', 'ACR_RATIO',
    'LBXSATSI', 'LBXSASSI', 'LBXSGTSI',
    'LBXWBCSI', 'LBXHCT', 'LBXHGB', 'LBXPLTSI',
]

LAB_FEATURES = [
    'LBXTC', 'LBDHDD', 'LBDLDL', 'LBXTR', 'TG_HDL_RATIO', 'NON_HDL_CHOL',
    'URXUMA', 'URXUCR', 'LBXSCR', 'ACR_RATIO',
    'LBXSATSI', 'LBXSASSI', 'LBXSGTSI',
    'LBXWBCSI', 'LBXHCT', 'LBXHGB', 'LBXPLTSI',
]

FEATURES_WITHOUT_LABS = [f for f in FEATURES_WITH_LABS if f not in LAB_FEATURES]


# =============================================================================
# TEST CASES - Example Individuals
# =============================================================================

def get_test_cases() -> Dict[str, Dict]:
    """
    Return example test cases that demonstrate the full capability of the predictor.

    These cases cover:
    1. Low Risk - Young, healthy individual
    2. High Risk - Older, obese with poor metabolic markers
    3. Borderline/Prediabetes - Middle-ground case
    4. Impact of Modifiable Factors - Same age/gender, different lifestyle
    5. Labs vs No Labs Comparison - Same individual, labs change prediction
    6. Edge Cases - Unusual but valid combinations
    """

    LBS_TO_KG = 0.453592

    test_cases = {
        # =====================================================================
        # CASE 1: Low Risk Individual
        # =====================================================================
        "Low Risk - Healthy Adult": {
            "description": "32-year-old active female with healthy BMI, good diet, no family history",
            "expected_outcome": "No Diabetes",
            "key_factors": ["Young age", "Healthy BMI (22)", "Active lifestyle", "No family history", "Excellent lab values"],
            "data": {
                # Demographics
                "RIDAGEYR": 32,
                "RIAGENDR": 2,  # Female
                # Anthropometric
                "BMXBMI": 22.5,
                "BMXWT": 62,  # kg
                "BMXHT": 166,  # cm
                "BMXWAIST": 72,  # cm
                "WAIST_HEIGHT_RATIO": 72/166,
                # Weight history (in lbs for raw, converted for derived)
                "WHD110": 135,  # 10 yrs ago
                "WHD120": 130,  # at age 25
                "WHD140": 140,  # max weight
                "WHD130": 30,   # age at heaviest
                "WEIGHT_CHANGE_10YR": 62 - (135 * LBS_TO_KG),
                "WEIGHT_CHANGE_25": 62 - (130 * LBS_TO_KG),
                "WEIGHT_FROM_MAX": (140 * LBS_TO_KG) - 62,
                # Blood pressure (3 readings)
                "BPXSY1": 112, "BPXSY2": 110, "BPXSY3": 108,
                "BPXDI1": 72, "BPXDI2": 70, "BPXDI3": 68,
                "AVG_SYS_BP": 110, "AVG_DIA_BP": 70,
                "PULSE_PRESSURE": 40, "MAP": 83.3, "BP_VARIABILITY": 2,
                # BP/Cholesterol questionnaire (1=Yes, 2=No)
                "BPQ020": 2,  # Never told high BP
                "BPQ040A": np.nan,  # Not applicable
                "BPQ080": 2,  # Never told high cholesterol
                "BPQ090D": np.nan,
                "BPQ100D": np.nan,
                # Dietary nutrients (24-hr recall)
                "DR1TKCAL": 1800,
                "DR1TPROT": 75,
                "DR1TCARB": 200,
                "DR1TSUGR": 45,
                "DR1TFIBE": 28,
                "DR1TTFAT": 65,
                "DR1TSFAT": 18,
                "DR1TMFAT": 25,
                "DR1TPFAT": 15,
                "DR1TSODI": 2000,
                "DR1TCAFF": 100,
                "DR1TALCO": 0,
                "DR1_320Z": 1200,
                "DR1_330Z": 400,
                "DR1BWATZ": 500,
                "TOTAL_WATER": 2100,
                "SAT_FAT_PCT": 27.7,
                "CARB_FIBER_RATIO": 7.1,
                "SUGAR_CARB_RATIO": 22.5,
                # Dietary behavior
                "DBQ700": 2,  # Very good diet
                "DBD895": 3,  # Few meals not home prepared
                "DBD900": 1,  # Rarely fast food
                "DBQ197": 3,  # Moderate milk
                # Alcohol (1=Never to higher=more frequent)
                "ALQ130": 1,  # 1 drink on drinking days
                "ALQ121": 3,  # 2-3x per month
                # Smoking (1=Yes, 2=No)
                "SMQ020": 2,  # Never smoked
                "SMQ040": np.nan,
                "SMD650": np.nan,
                # Physical activity (1=Yes, 2=No)
                "PAQ605": 2,  # No vigorous work
                "PAQ620": 1,  # Yes moderate work
                "PAQ635": 1,  # Yes walk/bicycle
                "PAQ650": 1,  # Yes vigorous recreation
                "PAQ665": 1,  # Yes moderate recreation
                "PAD680": 180,  # Sedentary min/day
                # Sleep
                "SLD012": 7.5,  # Weekday hours
                "SLD013": 8,    # Weekend hours
                "SLQ050": 2,    # No sleep disorder
                "SLQ310_HOURS": 6.5,
                "SLQ330_HOURS": 7.5,
                "WAKE_TIME_DIFF": 1,
                "SLEEP_DURATION_DIFF": 0.5,
                # Cardiovascular (1=Yes, 2=No)
                "CDQ010": 2,  # No shortness of breath
                # Medical history (1=Yes, 2=No)
                "MCQ160B": 2, "MCQ160C": 2, "MCQ160D": 2, "MCQ160E": 2, "MCQ160F": 2,
                "MCQ300C": 2,  # No family history diabetes
                "MCQ160L": 2,  # No liver condition
                "KIQ022": 2,   # No kidney issues
                "MCQ220": 2,   # No cancer
                "ANY_CVD": 0,
                # Depression (PHQ-9: 0=Not at all, 1=Several days, 2=More than half, 3=Nearly every day)
                "DPQ010": 0, "DPQ020": 0, "DPQ030": 0, "DPQ040": 0, "DPQ050": 0,
                "DPQ060": 0, "DPQ070": 0, "DPQ080": 0, "DPQ090": 0,
                "PHQ9_SCORE": 0,
                # Laboratory
                "LBXTC": 175,    # Total cholesterol
                "LBDHDD": 65,    # HDL
                "LBDLDL": 95,    # LDL
                "LBXTR": 75,     # Triglycerides
                "TG_HDL_RATIO": 1.15,
                "NON_HDL_CHOL": 110,
                "URXUMA": 5,     # Urine albumin
                "URXUCR": 120,   # Urine creatinine
                "LBXSCR": 0.9,   # Serum creatinine
                "ACR_RATIO": 0.04,
                "LBXSATSI": 20,  # ALT
                "LBXSASSI": 22,  # AST
                "LBXSGTSI": 18,  # GGT
                "LBXWBCSI": 6.5, # WBC
                "LBXHCT": 42,    # Hematocrit
                "LBXHGB": 14,    # Hemoglobin
                "LBXPLTSI": 250, # Platelets
            }
        },

        # =====================================================================
        # CASE 2: High Risk Individual
        # =====================================================================
        "High Risk - Metabolic Syndrome": {
            "description": "62-year-old male with obesity, hypertension, dyslipidemia, family history",
            "expected_outcome": "Diabetes",
            "key_factors": ["Older age (62)", "Obese BMI (34)", "Hypertension", "High TG/HDL ratio (5.0)", "Family history", "Sedentary"],
            "data": {
                # Demographics
                "RIDAGEYR": 62,
                "RIAGENDR": 1,  # Male
                # Anthropometric
                "BMXBMI": 34.2,
                "BMXWT": 105,  # kg
                "BMXHT": 175,  # cm
                "BMXWAIST": 112,  # cm - central obesity
                "WAIST_HEIGHT_RATIO": 112/175,
                # Weight history
                "WHD110": 210,  # 10 yrs ago (lbs)
                "WHD120": 165,  # at age 25 (lbs)
                "WHD140": 245,  # max weight (lbs)
                "WHD130": 55,   # age at heaviest
                "WEIGHT_CHANGE_10YR": 105 - (210 * LBS_TO_KG),
                "WEIGHT_CHANGE_25": 105 - (165 * LBS_TO_KG),
                "WEIGHT_FROM_MAX": (245 * LBS_TO_KG) - 105,
                # Blood pressure (elevated)
                "BPXSY1": 148, "BPXSY2": 145, "BPXSY3": 150,
                "BPXDI1": 92, "BPXDI2": 90, "BPXDI3": 88,
                "AVG_SYS_BP": 147.7, "AVG_DIA_BP": 90,
                "PULSE_PRESSURE": 57.7, "MAP": 109.2, "BP_VARIABILITY": 2.5,
                # BP/Cholesterol questionnaire
                "BPQ020": 1,  # Told high BP
                "BPQ040A": 1, # Taking BP meds
                "BPQ080": 1,  # Told high cholesterol
                "BPQ090D": 1, # Told to take cholesterol meds
                "BPQ100D": 1, # Currently taking
                # Dietary nutrients (poor diet)
                "DR1TKCAL": 2800,
                "DR1TPROT": 100,
                "DR1TCARB": 350,
                "DR1TSUGR": 120,
                "DR1TFIBE": 12,
                "DR1TTFAT": 110,
                "DR1TSFAT": 40,
                "DR1TMFAT": 35,
                "DR1TPFAT": 20,
                "DR1TSODI": 4500,
                "DR1TCAFF": 400,
                "DR1TALCO": 30,
                "DR1_320Z": 500,
                "DR1_330Z": 300,
                "DR1BWATZ": 0,
                "TOTAL_WATER": 800,
                "SAT_FAT_PCT": 36.4,
                "CARB_FIBER_RATIO": 29.2,
                "SUGAR_CARB_RATIO": 34.3,
                # Dietary behavior
                "DBQ700": 4,  # Poor diet
                "DBD895": 10, # Many meals out
                "DBD900": 8,  # Frequent fast food
                "DBQ197": 5,  # High milk
                # Alcohol
                "ALQ130": 3,
                "ALQ121": 5,  # 2-4x per week
                # Smoking
                "SMQ020": 1,  # Smoked 100+ cigs
                "SMQ040": 3,  # Former smoker
                "SMD650": np.nan,
                # Physical activity - sedentary
                "PAQ605": 2, "PAQ620": 2, "PAQ635": 2, "PAQ650": 2, "PAQ665": 2,
                "PAD680": 600,  # 10 hrs sedentary
                # Sleep
                "SLD012": 5.5,
                "SLD013": 7,
                "SLQ050": 1,  # Has sleep disorder
                "SLQ310_HOURS": 5.5,
                "SLQ330_HOURS": 8,
                "WAKE_TIME_DIFF": 2.5,
                "SLEEP_DURATION_DIFF": 1.5,
                # Cardiovascular
                "CDQ010": 1,  # Shortness of breath
                # Medical history
                "MCQ160B": 2, "MCQ160C": 1, "MCQ160D": 2, "MCQ160E": 2, "MCQ160F": 2,
                "MCQ300C": 1,  # Family history diabetes
                "MCQ160L": 2,
                "KIQ022": 2,
                "MCQ220": 2,
                "ANY_CVD": 1,
                # Depression
                "DPQ010": 1, "DPQ020": 1, "DPQ030": 1, "DPQ040": 0, "DPQ050": 0,
                "DPQ060": 0, "DPQ070": 0, "DPQ080": 0, "DPQ090": 0,
                "PHQ9_SCORE": 3,
                # Laboratory - poor values
                "LBXTC": 245,
                "LBDHDD": 38,
                "LBDLDL": 155,
                "LBXTR": 190,
                "TG_HDL_RATIO": 5.0,
                "NON_HDL_CHOL": 207,
                "URXUMA": 45,
                "URXUCR": 110,
                "LBXSCR": 1.2,
                "ACR_RATIO": 0.41,
                "LBXSATSI": 45,
                "LBXSASSI": 38,
                "LBXSGTSI": 55,
                "LBXWBCSI": 8.5,
                "LBXHCT": 46,
                "LBXHGB": 15.5,
                "LBXPLTSI": 220,
            }
        },

        # =====================================================================
        # CASE 3: Borderline/Prediabetes
        # =====================================================================
        "Borderline - Prediabetes Risk": {
            "description": "48-year-old female, overweight, moderately active, borderline labs",
            "expected_outcome": "Prediabetes",
            "key_factors": ["Middle age (48)", "Overweight BMI (28)", "Borderline lipids", "Moderate activity", "Some family history"],
            "data": {
                # Demographics
                "RIDAGEYR": 48,
                "RIAGENDR": 2,
                # Anthropometric
                "BMXBMI": 28.0,
                "BMXWT": 74,
                "BMXHT": 163,
                "BMXWAIST": 90,
                "WAIST_HEIGHT_RATIO": 90/163,
                # Weight history
                "WHD110": 155,
                "WHD120": 130,
                "WHD140": 175,
                "WHD130": 45,
                "WEIGHT_CHANGE_10YR": 74 - (155 * LBS_TO_KG),
                "WEIGHT_CHANGE_25": 74 - (130 * LBS_TO_KG),
                "WEIGHT_FROM_MAX": (175 * LBS_TO_KG) - 74,
                # Blood pressure (slightly elevated)
                "BPXSY1": 128, "BPXSY2": 126, "BPXSY3": 130,
                "BPXDI1": 82, "BPXDI2": 80, "BPXDI3": 78,
                "AVG_SYS_BP": 128, "AVG_DIA_BP": 80,
                "PULSE_PRESSURE": 48, "MAP": 96, "BP_VARIABILITY": 2,
                # BP/Cholesterol questionnaire
                "BPQ020": 2,
                "BPQ040A": np.nan,
                "BPQ080": 1,  # Told borderline cholesterol
                "BPQ090D": 2,
                "BPQ100D": np.nan,
                # Dietary nutrients (moderate)
                "DR1TKCAL": 2000,
                "DR1TPROT": 70,
                "DR1TCARB": 250,
                "DR1TSUGR": 70,
                "DR1TFIBE": 18,
                "DR1TTFAT": 75,
                "DR1TSFAT": 25,
                "DR1TMFAT": 28,
                "DR1TPFAT": 15,
                "DR1TSODI": 2800,
                "DR1TCAFF": 200,
                "DR1TALCO": 5,
                "DR1_320Z": 800,
                "DR1_330Z": 400,
                "DR1BWATZ": 200,
                "TOTAL_WATER": 1400,
                "SAT_FAT_PCT": 33.3,
                "CARB_FIBER_RATIO": 13.9,
                "SUGAR_CARB_RATIO": 28,
                # Dietary behavior
                "DBQ700": 3,  # Good
                "DBD895": 5,
                "DBD900": 3,
                "DBQ197": 3,
                # Alcohol
                "ALQ130": 1,
                "ALQ121": 4,
                # Smoking
                "SMQ020": 2,
                "SMQ040": np.nan,
                "SMD650": np.nan,
                # Physical activity - moderate
                "PAQ605": 2, "PAQ620": 1, "PAQ635": 1, "PAQ650": 2, "PAQ665": 1,
                "PAD680": 360,
                # Sleep
                "SLD012": 6.5,
                "SLD013": 7.5,
                "SLQ050": 2,
                "SLQ310_HOURS": 6,
                "SLQ330_HOURS": 7.5,
                "WAKE_TIME_DIFF": 1.5,
                "SLEEP_DURATION_DIFF": 1,
                # Cardiovascular
                "CDQ010": 2,
                # Medical history
                "MCQ160B": 2, "MCQ160C": 2, "MCQ160D": 2, "MCQ160E": 2, "MCQ160F": 2,
                "MCQ300C": 1,  # Family history
                "MCQ160L": 2,
                "KIQ022": 2,
                "MCQ220": 2,
                "ANY_CVD": 0,
                # Depression
                "DPQ010": 1, "DPQ020": 0, "DPQ030": 0, "DPQ040": 0, "DPQ050": 0,
                "DPQ060": 0, "DPQ070": 0, "DPQ080": 0, "DPQ090": 0,
                "PHQ9_SCORE": 1,
                # Laboratory - borderline
                "LBXTC": 210,
                "LBDHDD": 50,
                "LBDLDL": 130,
                "LBXTR": 140,
                "TG_HDL_RATIO": 2.8,
                "NON_HDL_CHOL": 160,
                "URXUMA": 20,
                "URXUCR": 100,
                "LBXSCR": 0.85,
                "ACR_RATIO": 0.2,
                "LBXSATSI": 28,
                "LBXSASSI": 26,
                "LBXSGTSI": 25,
                "LBXWBCSI": 7.2,
                "LBXHCT": 40,
                "LBXHGB": 13.5,
                "LBXPLTSI": 265,
            }
        },

        # =====================================================================
        # CASE 4: Lifestyle Impact Demonstration
        # =====================================================================
        "Lifestyle Matters - Active Despite Age": {
            "description": "58-year-old male, very active, excellent diet, normal weight despite older age",
            "expected_outcome": "No Diabetes (despite age)",
            "key_factors": ["Older age BUT active lifestyle", "Normal BMI (24)", "Excellent diet", "Good sleep", "Low TG/HDL ratio"],
            "data": {
                # Demographics
                "RIDAGEYR": 58,
                "RIAGENDR": 1,
                # Anthropometric
                "BMXBMI": 24.0,
                "BMXWT": 75,
                "BMXHT": 177,
                "BMXWAIST": 85,
                "WAIST_HEIGHT_RATIO": 85/177,
                # Weight history
                "WHD110": 165,
                "WHD120": 155,
                "WHD140": 175,
                "WHD130": 40,
                "WEIGHT_CHANGE_10YR": 75 - (165 * LBS_TO_KG),
                "WEIGHT_CHANGE_25": 75 - (155 * LBS_TO_KG),
                "WEIGHT_FROM_MAX": (175 * LBS_TO_KG) - 75,
                # Blood pressure (normal)
                "BPXSY1": 118, "BPXSY2": 116, "BPXSY3": 120,
                "BPXDI1": 74, "BPXDI2": 72, "BPXDI3": 74,
                "AVG_SYS_BP": 118, "AVG_DIA_BP": 73.3,
                "PULSE_PRESSURE": 44.7, "MAP": 88.2, "BP_VARIABILITY": 2,
                # BP/Cholesterol questionnaire
                "BPQ020": 2, "BPQ040A": np.nan, "BPQ080": 2, "BPQ090D": np.nan, "BPQ100D": np.nan,
                # Dietary nutrients (excellent)
                "DR1TKCAL": 2200,
                "DR1TPROT": 100,
                "DR1TCARB": 260,
                "DR1TSUGR": 50,
                "DR1TFIBE": 35,
                "DR1TTFAT": 70,
                "DR1TSFAT": 18,
                "DR1TMFAT": 30,
                "DR1TPFAT": 18,
                "DR1TSODI": 2200,
                "DR1TCAFF": 150,
                "DR1TALCO": 10,
                "DR1_320Z": 1500,
                "DR1_330Z": 500,
                "DR1BWATZ": 300,
                "TOTAL_WATER": 2300,
                "SAT_FAT_PCT": 25.7,
                "CARB_FIBER_RATIO": 7.4,
                "SUGAR_CARB_RATIO": 19.2,
                # Dietary behavior
                "DBQ700": 1,  # Excellent
                "DBD895": 2,
                "DBD900": 0,
                "DBQ197": 3,
                # Alcohol (moderate)
                "ALQ130": 1,
                "ALQ121": 3,
                # Smoking
                "SMQ020": 2, "SMQ040": np.nan, "SMD650": np.nan,
                # Physical activity - VERY active
                "PAQ605": 1, "PAQ620": 1, "PAQ635": 1, "PAQ650": 1, "PAQ665": 1,
                "PAD680": 120,
                # Sleep
                "SLD012": 7,
                "SLD013": 7.5,
                "SLQ050": 2,
                "SLQ310_HOURS": 6,
                "SLQ330_HOURS": 6.5,
                "WAKE_TIME_DIFF": 0.5,
                "SLEEP_DURATION_DIFF": 0.5,
                # Cardiovascular
                "CDQ010": 2,
                # Medical history
                "MCQ160B": 2, "MCQ160C": 2, "MCQ160D": 2, "MCQ160E": 2, "MCQ160F": 2,
                "MCQ300C": 2,
                "MCQ160L": 2, "KIQ022": 2, "MCQ220": 2,
                "ANY_CVD": 0,
                # Depression
                "DPQ010": 0, "DPQ020": 0, "DPQ030": 0, "DPQ040": 0, "DPQ050": 0,
                "DPQ060": 0, "DPQ070": 0, "DPQ080": 0, "DPQ090": 0,
                "PHQ9_SCORE": 0,
                # Laboratory - excellent
                "LBXTC": 185,
                "LBDHDD": 62,
                "LBDLDL": 100,
                "LBXTR": 85,
                "TG_HDL_RATIO": 1.37,
                "NON_HDL_CHOL": 123,
                "URXUMA": 8,
                "URXUCR": 130,
                "LBXSCR": 1.0,
                "ACR_RATIO": 0.06,
                "LBXSATSI": 22,
                "LBXSASSI": 24,
                "LBXSGTSI": 20,
                "LBXWBCSI": 5.8,
                "LBXHCT": 44,
                "LBXHGB": 15,
                "LBXPLTSI": 230,
            }
        },

        # =====================================================================
        # CASE 5: Young with Poor Lifestyle
        # =====================================================================
        "Young but High Risk - Poor Lifestyle": {
            "description": "28-year-old male, obese, sedentary, poor diet, family history",
            "expected_outcome": "Prediabetes (despite young age)",
            "key_factors": ["Young age BUT obese", "Sedentary", "Poor diet", "Family history", "Elevated TG/HDL"],
            "data": {
                # Demographics
                "RIDAGEYR": 28,
                "RIAGENDR": 1,
                # Anthropometric - obese
                "BMXBMI": 35.5,
                "BMXWT": 115,
                "BMXHT": 180,
                "BMXWAIST": 115,
                "WAIST_HEIGHT_RATIO": 115/180,
                # Weight history
                "WHD110": 230,
                "WHD120": 180,
                "WHD140": 260,
                "WHD130": 27,
                "WEIGHT_CHANGE_10YR": 115 - (230 * LBS_TO_KG),
                "WEIGHT_CHANGE_25": 115 - (180 * LBS_TO_KG),
                "WEIGHT_FROM_MAX": (260 * LBS_TO_KG) - 115,
                # Blood pressure
                "BPXSY1": 135, "BPXSY2": 138, "BPXSY3": 132,
                "BPXDI1": 85, "BPXDI2": 88, "BPXDI3": 84,
                "AVG_SYS_BP": 135, "AVG_DIA_BP": 85.7,
                "PULSE_PRESSURE": 49.3, "MAP": 102.1, "BP_VARIABILITY": 3,
                # BP/Cholesterol
                "BPQ020": 2, "BPQ040A": np.nan, "BPQ080": 2, "BPQ090D": np.nan, "BPQ100D": np.nan,
                # Dietary nutrients (very poor)
                "DR1TKCAL": 3500,
                "DR1TPROT": 110,
                "DR1TCARB": 450,
                "DR1TSUGR": 180,
                "DR1TFIBE": 10,
                "DR1TTFAT": 140,
                "DR1TSFAT": 55,
                "DR1TMFAT": 45,
                "DR1TPFAT": 25,
                "DR1TSODI": 5000,
                "DR1TCAFF": 350,
                "DR1TALCO": 40,
                "DR1_320Z": 200,
                "DR1_330Z": 100,
                "DR1BWATZ": 0,
                "TOTAL_WATER": 300,
                "SAT_FAT_PCT": 39.3,
                "CARB_FIBER_RATIO": 45,
                "SUGAR_CARB_RATIO": 40,
                # Dietary behavior
                "DBQ700": 5,  # Very poor
                "DBD895": 15,
                "DBD900": 12,
                "DBQ197": 6,
                # Alcohol (heavy)
                "ALQ130": 5,
                "ALQ121": 6,
                # Smoking
                "SMQ020": 1, "SMQ040": 1, "SMD650": 10,
                # Physical activity - none
                "PAQ605": 2, "PAQ620": 2, "PAQ635": 2, "PAQ650": 2, "PAQ665": 2,
                "PAD680": 720,
                # Sleep (poor)
                "SLD012": 5,
                "SLD013": 9,
                "SLQ050": 1,
                "SLQ310_HOURS": 7,
                "SLQ330_HOURS": 11,
                "WAKE_TIME_DIFF": 4,
                "SLEEP_DURATION_DIFF": 4,
                # Cardiovascular
                "CDQ010": 1,
                # Medical history
                "MCQ160B": 2, "MCQ160C": 2, "MCQ160D": 2, "MCQ160E": 2, "MCQ160F": 2,
                "MCQ300C": 1,  # Family history
                "MCQ160L": 2, "KIQ022": 2, "MCQ220": 2,
                "ANY_CVD": 0,
                # Depression
                "DPQ010": 2, "DPQ020": 2, "DPQ030": 1, "DPQ040": 1, "DPQ050": 0,
                "DPQ060": 0, "DPQ070": 0, "DPQ080": 0, "DPQ090": 0,
                "PHQ9_SCORE": 6,
                # Laboratory - concerning
                "LBXTC": 225,
                "LBDHDD": 35,
                "LBDLDL": 140,
                "LBXTR": 200,
                "TG_HDL_RATIO": 5.7,
                "NON_HDL_CHOL": 190,
                "URXUMA": 30,
                "URXUCR": 95,
                "LBXSCR": 1.0,
                "ACR_RATIO": 0.32,
                "LBXSATSI": 55,
                "LBXSASSI": 45,
                "LBXSGTSI": 75,
                "LBXWBCSI": 9.5,
                "LBXHCT": 48,
                "LBXHGB": 16,
                "LBXPLTSI": 280,
            }
        },

        # =====================================================================
        # CASE 6: Lab Values Make the Difference
        # =====================================================================
        "Labs Critical - Hidden Risk": {
            "description": "45-year-old with normal appearance but very poor lab values revealing hidden metabolic dysfunction",
            "expected_outcome": "Prediabetes/Diabetes (labs reveal hidden risk)",
            "key_factors": ["Normal BMI", "Normal BP", "BUT: Very high TG/HDL (6.5)", "Elevated liver enzymes", "Microalbuminuria"],
            "data": {
                # Demographics
                "RIDAGEYR": 45,
                "RIAGENDR": 1,
                # Anthropometric - appears normal
                "BMXBMI": 26.0,
                "BMXWT": 80,
                "BMXHT": 175,
                "BMXWAIST": 92,
                "WAIST_HEIGHT_RATIO": 92/175,
                # Weight history
                "WHD110": 175,
                "WHD120": 155,
                "WHD140": 185,
                "WHD130": 40,
                "WEIGHT_CHANGE_10YR": 80 - (175 * LBS_TO_KG),
                "WEIGHT_CHANGE_25": 80 - (155 * LBS_TO_KG),
                "WEIGHT_FROM_MAX": (185 * LBS_TO_KG) - 80,
                # Blood pressure - borderline normal
                "BPXSY1": 122, "BPXSY2": 124, "BPXSY3": 120,
                "BPXDI1": 78, "BPXDI2": 80, "BPXDI3": 76,
                "AVG_SYS_BP": 122, "AVG_DIA_BP": 78,
                "PULSE_PRESSURE": 44, "MAP": 92.7, "BP_VARIABILITY": 2,
                # BP/Cholesterol
                "BPQ020": 2, "BPQ040A": np.nan, "BPQ080": 1, "BPQ090D": 2, "BPQ100D": np.nan,
                # Dietary nutrients
                "DR1TKCAL": 2400,
                "DR1TPROT": 90,
                "DR1TCARB": 300,
                "DR1TSUGR": 95,
                "DR1TFIBE": 15,
                "DR1TTFAT": 90,
                "DR1TSFAT": 32,
                "DR1TMFAT": 32,
                "DR1TPFAT": 18,
                "DR1TSODI": 3200,
                "DR1TCAFF": 250,
                "DR1TALCO": 20,
                "DR1_320Z": 600,
                "DR1_330Z": 300,
                "DR1BWATZ": 0,
                "TOTAL_WATER": 900,
                "SAT_FAT_PCT": 35.6,
                "CARB_FIBER_RATIO": 20,
                "SUGAR_CARB_RATIO": 31.7,
                # Dietary behavior
                "DBQ700": 3,
                "DBD895": 8,
                "DBD900": 5,
                "DBQ197": 4,
                # Alcohol
                "ALQ130": 2,
                "ALQ121": 5,
                # Smoking
                "SMQ020": 2, "SMQ040": np.nan, "SMD650": np.nan,
                # Physical activity - light
                "PAQ605": 2, "PAQ620": 2, "PAQ635": 1, "PAQ650": 2, "PAQ665": 2,
                "PAD680": 420,
                # Sleep
                "SLD012": 6,
                "SLD013": 7,
                "SLQ050": 2,
                "SLQ310_HOURS": 6.5,
                "SLQ330_HOURS": 8,
                "WAKE_TIME_DIFF": 1.5,
                "SLEEP_DURATION_DIFF": 1,
                # Cardiovascular
                "CDQ010": 2,
                # Medical history
                "MCQ160B": 2, "MCQ160C": 2, "MCQ160D": 2, "MCQ160E": 2, "MCQ160F": 2,
                "MCQ300C": 1,
                "MCQ160L": 2, "KIQ022": 2, "MCQ220": 2,
                "ANY_CVD": 0,
                # Depression
                "DPQ010": 0, "DPQ020": 0, "DPQ030": 0, "DPQ040": 0, "DPQ050": 0,
                "DPQ060": 0, "DPQ070": 0, "DPQ080": 0, "DPQ090": 0,
                "PHQ9_SCORE": 0,
                # Laboratory - VERY CONCERNING (hidden)
                "LBXTC": 260,
                "LBDHDD": 32,  # Very low HDL
                "LBDLDL": 165,
                "LBXTR": 210,  # High triglycerides
                "TG_HDL_RATIO": 6.5,  # Very high - indicates severe insulin resistance
                "NON_HDL_CHOL": 228,
                "URXUMA": 55,  # Microalbuminuria
                "URXUCR": 100,
                "LBXSCR": 1.1,
                "ACR_RATIO": 0.55,  # Elevated
                "LBXSATSI": 65,  # Elevated ALT
                "LBXSASSI": 48,
                "LBXSGTSI": 85,  # Elevated GGT
                "LBXWBCSI": 8.8,
                "LBXHCT": 47,
                "LBXHGB": 16,
                "LBXPLTSI": 290,
            }
        },
    }

    return test_cases


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def prepare_input_data(user_data: Dict, feature_order: List[str]) -> pd.DataFrame:
    """Prepare user input data for model prediction using the exact feature order."""
    # Create DataFrame with expected feature order
    df = pd.DataFrame([user_data])

    # Ensure all features are present (use NaN for missing)
    for feat in feature_order:
        if feat not in df.columns:
            df[feat] = np.nan

    # Select only required features in correct order
    df = df[feature_order]

    return df


def predict_diabetes_risk(models: Dict, user_data: Dict, feature_order: Dict, use_labs: bool = True) -> Dict:
    """
    Make diabetes risk prediction.

    Returns dict with:
    - predicted_class: 0/1/2
    - class_label: string
    - probabilities: dict of class probabilities
    - hba1c_prediction: predicted HbA1c (if regression model available)
    """
    # Select appropriate model
    cls_model_key = 'classification_with_labs' if use_labs else 'classification_without_labs'
    reg_model_key = 'regression_with_labs' if use_labs else 'regression_without_labs'
    feature_key = 'with_labs' if use_labs else 'without_labs'

    if cls_model_key not in models:
        return {"error": f"Model not found: {cls_model_key}"}

    if feature_key not in feature_order:
        return {"error": f"Feature order not found: {feature_key}"}

    cls_model = models[cls_model_key]
    reg_model = models.get(reg_model_key)
    features = feature_order[feature_key]

    # Prepare input with exact feature order from training data
    X = prepare_input_data(user_data, features)

    # Classification prediction
    predicted_class = int(cls_model.predict(X)[0])
    probabilities = cls_model.predict_proba(X)[0]

    class_labels = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}

    result = {
        "predicted_class": predicted_class,
        "class_label": class_labels[predicted_class],
        "probabilities": {
            "No Diabetes": float(probabilities[0]),
            "Prediabetes": float(probabilities[1]),
            "Diabetes": float(probabilities[2]),
        },
        "confidence": float(max(probabilities)),
        "model_used": "with labs" if use_labs else "without labs",
    }

    # Regression prediction (HbA1c)
    if reg_model is not None:
        # Use same feature order as classification model
        hba1c_pred = float(reg_model.predict(X)[0])
        result["hba1c_prediction"] = hba1c_pred

        # HbA1c interpretation
        if hba1c_pred < 5.7:
            result["hba1c_interpretation"] = "Normal"
        elif hba1c_pred < 6.5:
            result["hba1c_interpretation"] = "Prediabetes range"
        else:
            result["hba1c_interpretation"] = "Diabetes range"

    return result


# =============================================================================
# STREAMLIT PAGES
# =============================================================================

def page_risk_calculator():
    """Risk Calculator page - main prediction interface."""
    st.title("Diabetes Risk Calculator")
    st.markdown("Enter health metrics to predict diabetes risk using our trained LightGBM model.")

    # Load models and feature order
    models = load_models()
    feature_order = load_feature_order()

    if not models:
        st.error("No models found. Please ensure models are trained and saved in models/advanced/")
        return

    if not feature_order:
        st.error("Feature order not found. Please ensure processed data exists in data/processed/")
        return

    # Sidebar options
    st.sidebar.header("Settings")
    use_labs = st.sidebar.checkbox("Include laboratory values", value=True,
                                    help="If unchecked, prediction uses only questionnaire and exam data")

    # Test case selection
    st.sidebar.header("Quick Start")
    test_cases = get_test_cases()
    case_names = ["-- Select a test case --"] + list(test_cases.keys())
    selected_case = st.sidebar.selectbox("Load example individual", case_names)

    if selected_case != "-- Select a test case --":
        case = test_cases[selected_case]
        st.info(f"**{selected_case}**: {case['description']}")
        st.markdown(f"**Expected outcome**: {case['expected_outcome']}")
        st.markdown(f"**Key factors**: {', '.join(case['key_factors'])}")

        user_data = case['data']

        # Make prediction
        with st.spinner("Calculating risk..."):
            result = predict_diabetes_risk(models, user_data, feature_order, use_labs)

        # Display results
        st.header("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Color based on prediction
            colors = {0: "green", 1: "orange", 2: "red"}
            color = colors[result['predicted_class']]
            st.markdown(f"### Predicted: <span style='color:{color}'>{result['class_label']}</span>",
                       unsafe_allow_html=True)
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")

        with col2:
            st.markdown("### Class Probabilities")
            for cls, prob in result['probabilities'].items():
                st.progress(prob, text=f"{cls}: {prob*100:.1f}%")

        with col3:
            if 'hba1c_prediction' in result:
                st.markdown("### Predicted HbA1c")
                st.metric("HbA1c", f"{result['hba1c_prediction']:.2f}%")
                st.caption(f"Interpretation: {result['hba1c_interpretation']}")

        # Show comparison: with labs vs without labs
        if use_labs:
            st.markdown("---")
            st.subheader("Comparison: With Labs vs Without Labs")

            result_no_labs = predict_diabetes_risk(models, user_data, feature_order, use_labs=False)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**With Laboratory Values**")
                st.write(f"Prediction: {result['class_label']}")
                st.write(f"Confidence: {result['confidence']*100:.1f}%")

            with col2:
                st.markdown("**Without Laboratory Values**")
                st.write(f"Prediction: {result_no_labs['class_label']}")
                st.write(f"Confidence: {result_no_labs['confidence']*100:.1f}%")

            if result['predicted_class'] != result_no_labs['predicted_class']:
                st.warning("Lab values changed the prediction! This demonstrates the value of blood tests in risk assessment.")

    else:
        st.markdown("""
        ### How to Use
        1. Select an example individual from the sidebar to see a demonstration
        2. Or enter your own values below for a custom prediction

        The model uses NHANES survey data to predict diabetes risk based on:
        - Demographics (age, gender)
        - Body measurements (BMI, waist circumference)
        - Blood pressure readings
        - Dietary patterns
        - Physical activity and lifestyle
        - Medical history
        - Laboratory values (optional)
        """)


def page_test_cases():
    """Display all test cases with predictions."""
    st.title("Example Test Cases")
    st.markdown("""
    These example individuals demonstrate the full capability of the diabetes prediction model.
    Each case highlights different risk factors and shows how the model responds to various profiles.
    """)

    models = load_models()
    feature_order = load_feature_order()

    if not models:
        st.error("Models not loaded.")
        return

    if not feature_order:
        st.error("Feature order not found.")
        return

    test_cases = get_test_cases()

    for name, case in test_cases.items():
        with st.expander(f"{name}", expanded=False):
            st.markdown(f"**Description**: {case['description']}")
            st.markdown(f"**Expected Outcome**: {case['expected_outcome']}")
            st.markdown(f"**Key Factors**: {', '.join(case['key_factors'])}")

            col1, col2 = st.columns(2)

            # Prediction with labs
            result_with = predict_diabetes_risk(models, case['data'], feature_order, use_labs=True)

            with col1:
                st.markdown("#### With Laboratory Values")
                colors = {"No Diabetes": "green", "Prediabetes": "orange", "Diabetes": "red"}
                color = colors[result_with['class_label']]
                st.markdown(f"**Prediction**: <span style='color:{color}'>{result_with['class_label']}</span>",
                           unsafe_allow_html=True)
                st.write(f"Confidence: {result_with['confidence']*100:.1f}%")
                if 'hba1c_prediction' in result_with:
                    st.write(f"Predicted HbA1c: {result_with['hba1c_prediction']:.2f}%")

            # Prediction without labs
            result_without = predict_diabetes_risk(models, case['data'], feature_order, use_labs=False)

            with col2:
                st.markdown("#### Without Laboratory Values")
                color = colors[result_without['class_label']]
                st.markdown(f"**Prediction**: <span style='color:{color}'>{result_without['class_label']}</span>",
                           unsafe_allow_html=True)
                st.write(f"Confidence: {result_without['confidence']*100:.1f}%")

            # Check if prediction matches expected
            expected_class = {"No Diabetes": 0, "Prediabetes": 1, "Diabetes": 2,
                             "No Diabetes (despite age)": 0, "Prediabetes (despite young age)": 1,
                             "Prediabetes/Diabetes (labs reveal hidden risk)": 1}

            # Show all probabilities
            st.markdown("---")
            st.markdown("**Probability Distribution (With Labs)**")
            prob_data = pd.DataFrame({
                'Class': list(result_with['probabilities'].keys()),
                'Probability': list(result_with['probabilities'].values())
            })
            st.bar_chart(prob_data.set_index('Class'))


def page_model_info():
    """Model Information page."""
    st.title("Model Information")

    st.markdown("""
    ## About This Model

    This diabetes risk prediction tool uses **LightGBM** (Light Gradient Boosting Machine),
    trained on data from the National Health and Nutrition Examination Survey (NHANES) 2015-2018.

    ### Key Features

    | Aspect | Details |
    |--------|---------|
    | **Algorithm** | LightGBM (gradient boosting) |
    | **Training Data** | NHANES 2015-2018 (11,698 adults) |
    | **Classes** | No Diabetes, Prediabetes, Diabetes |
    | **Features** | 109 (with labs) or 92 (without labs) |

    ### Performance Metrics (Test Set)

    | Model | F1 Macro | ROC AUC | Accuracy |
    |-------|----------|---------|----------|
    | **With Labs** | 0.612 | 0.816 | 63.0% |
    | **Without Labs** | 0.549 | 0.756 | 56.5% |

    ### Feature Categories

    The model considers multiple health dimensions:

    1. **Demographics**: Age, gender
    2. **Anthropometric**: BMI, waist circumference, waist-to-height ratio
    3. **Weight History**: Weight changes over time
    4. **Blood Pressure**: Multiple readings + derived metrics
    5. **Dietary Patterns**: Nutrient intake, diet quality
    6. **Lifestyle**: Physical activity, smoking, alcohol, sleep
    7. **Medical History**: Cardiovascular disease, family history
    8. **Mental Health**: PHQ-9 depression screening
    9. **Laboratory Values** (optional): Lipids, kidney/liver function, blood count

    ### Important Limitations

    - This is a **screening tool**, not a diagnostic device
    - Predictions are probabilistic estimates, not definitive diagnoses
    - Always consult a healthcare provider for medical decisions
    - Model trained on US population; may vary for other populations
    - Self-reported data (diet, activity) may have recall bias

    ### Data Source

    [NHANES](https://www.cdc.gov/nchs/nhanes/index.htm) is a nationally representative
    survey conducted by the CDC to assess the health and nutritional status of adults
    and children in the United States.
    """)


def page_population_insights():
    """Population Insights page."""
    st.title("Population Insights")

    st.markdown("""
    ## Key Findings from NHANES Analysis

    ### Diabetes Prevalence

    In the study population (US adults, 2015-2018):

    | Status | Percentage |
    |--------|------------|
    | No Diabetes | 48.6% |
    | Prediabetes | 32.4% |
    | Diabetes | 19.0% |

    ### Top Risk Factors (SHAP Analysis)

    These factors most strongly influence the model's predictions:

    1. **Age** - Strongest predictor (non-modifiable)
    2. **TG/HDL Ratio** - Insulin resistance marker
    3. **Waist-to-Height Ratio** - Central obesity measure
    4. **BMI** - Overall obesity
    5. **GGT (Liver Enzyme)** - Metabolic stress marker
    6. **Age at Heaviest Weight** - Weight history
    7. **Family History** - Genetic predisposition
    8. **Systolic Blood Pressure** - Cardiovascular health
    9. **Triglycerides** - Lipid metabolism
    10. **Shortness of Breath** - Cardiovascular capacity

    ### Modifiable vs Non-Modifiable Factors

    **Good News**: ~30% of prediction importance comes from **modifiable factors**:
    - Weight management (BMI, waist circumference)
    - Blood pressure control
    - Diet quality (fiber, sugar intake)
    - Physical activity
    - Sleep patterns

    **Non-Modifiable** (~25% importance):
    - Age
    - Family history
    - Gender

    **Laboratory Values** (~35% importance):
    - Most are influenced by lifestyle
    - Early detection enables intervention

    ### Actionable Insights

    Based on the analysis, key prevention strategies include:

    1. **Maintain healthy weight** - BMI < 25, waist < half height
    2. **Stay active** - Regular physical activity
    3. **Improve diet quality** - More fiber, less refined sugar
    4. **Monitor blood pressure** - Keep < 130/80
    5. **Regular screening** - Especially after age 40 or with family history
    6. **Prioritize sleep** - Consistent 7-8 hours
    """)


def page_compare_scenarios():
    """Compare Scenarios page - What-if analysis."""
    st.title("Compare Scenarios")
    st.markdown("See how changing risk factors affects diabetes risk prediction.")

    models = load_models()
    feature_order = load_feature_order()

    if not models:
        st.error("Models not loaded.")
        return

    if not feature_order:
        st.error("Feature order not found.")
        return

    st.markdown("### Select two scenarios to compare")

    test_cases = get_test_cases()
    case_names = list(test_cases.keys())

    col1, col2 = st.columns(2)

    with col1:
        scenario1 = st.selectbox("Scenario 1", case_names, index=0)

    with col2:
        scenario2 = st.selectbox("Scenario 2", case_names, index=1)

    if scenario1 and scenario2:
        case1 = test_cases[scenario1]
        case2 = test_cases[scenario2]

        result1 = predict_diabetes_risk(models, case1['data'], feature_order, use_labs=True)
        result2 = predict_diabetes_risk(models, case2['data'], feature_order, use_labs=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {scenario1}")
            st.markdown(f"*{case1['description']}*")
            colors = {"No Diabetes": "green", "Prediabetes": "orange", "Diabetes": "red"}
            color = colors[result1['class_label']]
            st.markdown(f"**Prediction**: <span style='color:{color};font-size:24px'>{result1['class_label']}</span>",
                       unsafe_allow_html=True)
            st.metric("Confidence", f"{result1['confidence']*100:.1f}%")

            st.markdown("**Key Factors:**")
            for factor in case1['key_factors']:
                st.write(f"- {factor}")

        with col2:
            st.markdown(f"### {scenario2}")
            st.markdown(f"*{case2['description']}*")
            color = colors[result2['class_label']]
            st.markdown(f"**Prediction**: <span style='color:{color};font-size:24px'>{result2['class_label']}</span>",
                       unsafe_allow_html=True)
            st.metric("Confidence", f"{result2['confidence']*100:.1f}%")

            st.markdown("**Key Factors:**")
            for factor in case2['key_factors']:
                st.write(f"- {factor}")

        # Show key differences
        st.markdown("---")
        st.markdown("### Key Metric Differences")

        metrics_to_compare = [
            ("Age", "RIDAGEYR", "years"),
            ("BMI", "BMXBMI", "kg/m²"),
            ("Waist Circumference", "BMXWAIST", "cm"),
            ("Systolic BP", "AVG_SYS_BP", "mmHg"),
            ("TG/HDL Ratio", "TG_HDL_RATIO", ""),
            ("PHQ-9 Score", "PHQ9_SCORE", ""),
            ("Sedentary Time", "PAD680", "min/day"),
        ]

        comparison_data = []
        for label, var, unit in metrics_to_compare:
            val1 = case1['data'].get(var, np.nan)
            val2 = case2['data'].get(var, np.nan)
            if not pd.isna(val1) and not pd.isna(val2):
                comparison_data.append({
                    "Metric": label,
                    scenario1: f"{val1:.1f} {unit}",
                    scenario2: f"{val2:.1f} {unit}",
                })

        if comparison_data:
            st.table(pd.DataFrame(comparison_data))


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""

    # Sidebar navigation
    st.sidebar.title("Navigation")

    pages = {
        "Risk Calculator": page_risk_calculator,
        "Example Test Cases": page_test_cases,
        "Compare Scenarios": page_compare_scenarios,
        "Population Insights": page_population_insights,
        "Model Information": page_model_info,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Run selected page
    pages[selection]()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **NHANES Diabetes Predictor**
    Built with Streamlit
    Data: NHANES 2015-2018
    """)


if __name__ == "__main__":
    main()
