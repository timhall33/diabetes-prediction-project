# NHANES Data Dictionary

This document describes all variables used in the diabetes prediction project.

## Overview

Data source: [NHANES (National Health and Nutrition Examination Survey)](https://www.cdc.gov/nchs/nhanes/)

Survey years: 1999-2018 (10 two-year cycles)

---

## Target Variables

### Classification Target (3-class)

| Class | Value | Criteria |
|-------|-------|----------|
| No Diabetes | 0 | Does not meet diabetes or prediabetes criteria |
| Prediabetes | 1 | HbA1c 5.7-6.4% OR Fasting glucose 100-125 mg/dL (not diabetic) |
| Diabetes | 2 | DIQ010=1 (told have diabetes) OR HbA1c >= 6.5% OR Fasting glucose >= 126 mg/dL OR DIQ070=1 (taking insulin) OR DIQ050=1 (taking diabetes pills) |

### Regression Target

| Variable | Description | Units |
|----------|-------------|-------|
| LBXGH | Glycohemoglobin (HbA1c) | % |

---

## Variable Categories

*To be populated during Phase 1-2 as data is downloaded and explored.*

### Demographics
| Variable | Description | Type | Valid Range |
|----------|-------------|------|-------------|
| RIDAGEYR | Age at screening | Continuous | 0-80 (capped) |
| RIAGENDR | Gender | Categorical | 1=Male, 2=Female |

### Anthropometric
*TBD*

### Laboratory
*TBD*

### Questionnaire
*TBD*

---

## Variable Harmonization

Some variables changed names across survey years. This section documents the mappings.

| Modern Variable | Legacy Variable(s) | Years | Notes |
|-----------------|-------------------|-------|-------|
| ALQ121 | ALQ120Q + ALQ120U | Pre-2017 | Alcohol frequency |
| *More to be added* | | | |

---

## Missing Value Codes

NHANES uses specific codes for missing data:

| Code | Meaning |
|------|---------|
| . | Missing |
| 7 | Refused |
| 9 | Don't know |
| 77 | Refused (2-digit) |
| 99 | Don't know (2-digit) |
| 777 | Refused (3-digit) |
| 999 | Don't know (3-digit) |

---

## References

- [NHANES Documentation](https://wwwn.cdc.gov/nchs/nhanes/)
- [ADA Diabetes Diagnosis Criteria](https://diabetes.org/about-diabetes/diagnosis)
