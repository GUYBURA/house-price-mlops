import pytest

from src import config
from src.data import clean_data
from src.features import add_features, split_target


def test_total_bath_counts_half_baths_as_half(raw):
    row = add_features(clean_data(raw)).iloc[0]
    src = raw.iloc[0]
    expected = (src["Full Bath"] + 0.5 * src["Half Bath"]
                + src["Bsmt Full Bath"] + 0.5 * src["Bsmt Half Bath"])
    assert row["total_bath"] == expected


def test_age_at_sale_is_never_far_negative(raw):
    features = add_features(clean_data(raw))
    # one Ames house is sold a year before its build year is recorded; anything lower is a bug
    assert features["age_at_sale"].min() >= -1


def test_quality_columns_become_ordered_numbers(raw):
    features = add_features(clean_data(raw))
    for col in config.QUAL_COLS:
        assert features[col].dtype.kind in "iu", f"{col} is not integer"
        assert features[col].isna().sum() == 0, f"{col} has unmapped values"
        assert features[col].between(0, 5).all()


def test_no_basement_maps_to_zero(raw):
    no_basement = raw["Bsmt Qual"].isna()
    features = add_features(clean_data(raw))
    assert (features.loc[no_basement, "Bsmt Qual"] == 0).all()


def test_drop_columns_are_removed(raw):
    features = add_features(clean_data(raw))
    assert not [c for c in config.DROP_COLS if c in features.columns]


def test_add_features_works_without_id_columns(raw):
    """New houses do not carry this CSV's Order/PID."""
    features = add_features(clean_data(raw.drop(columns=["Order", "PID"])))
    assert len(features) == len(raw)


def test_split_target_removes_target_from_x(train_xy):
    X, y = train_xy
    assert config.TARGET not in X.columns
    assert y.name == config.TARGET
    assert len(X) == len(y)


def test_split_target_requires_target(raw):
    features = add_features(clean_data(raw.drop(columns=config.TARGET)))
    with pytest.raises(KeyError):
        split_target(features)


def test_baseline_features_exist(train_xy):
    X, _ = train_xy
    assert not [c for c in config.BASELINE_FEATURES if c not in X.columns]
