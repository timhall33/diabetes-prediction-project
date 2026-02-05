import numpy as np
import pandas as pd

from src.data.cleaners import clean_target_columns


def test_clean_target_columns_only_questionnaire():
    df = pd.DataFrame({
        "DIQ010": [1, 7, 9, 2],
        "LBXGH": [7.0, 5.5, np.nan, 9.0],
    })

    cleaned, _ = clean_target_columns(df)

    assert np.isnan(cleaned.loc[1, "DIQ010"])
    assert np.isnan(cleaned.loc[2, "DIQ010"])
    assert cleaned.loc[0, "DIQ010"] == 1
    assert cleaned.loc[3, "DIQ010"] == 2

    # Lab values should remain unchanged
    assert cleaned.loc[0, "LBXGH"] == 7.0
    assert cleaned.loc[3, "LBXGH"] == 9.0
