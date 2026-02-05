from src.data.validators import check_target_leakage


def test_target_leakage_detection():
    features = ["RIDAGEYR", "DIQ010", "BMXBMI"]
    leaked = check_target_leakage(features)
    assert "DIQ010" in leaked
