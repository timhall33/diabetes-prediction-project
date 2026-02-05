from src.features.builders import get_feature_sets


def test_feature_sets_consistency():
    feature_sets = get_feature_sets()

    with_labs = feature_sets["with_labs"]
    without_labs = feature_sets["without_labs"]
    lab_only = set(feature_sets["lab_only"]["features"])

    assert with_labs["n_features"] == len(with_labs["features"])
    assert without_labs["n_features"] == len(without_labs["features"])
    assert with_labs["n_features"] > without_labs["n_features"]
    assert not lab_only.intersection(without_labs["features"])
