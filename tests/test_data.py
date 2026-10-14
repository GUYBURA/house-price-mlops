from src import config
from src.data import clean_data, remove_outliers


def test_load_data_has_expected_columns(raw):
    assert len(raw) == 2930
    assert config.TARGET in raw.columns
    for col in config.NONE_FILL_COLS + config.ZERO_FILL_COLS + config.QUAL_COLS:
        assert col in raw.columns, f"{col} missing from CSV"


def test_clean_data_fills_missing_values(raw):
    cleaned = clean_data(raw)
    assert cleaned[config.NONE_FILL_COLS].isna().sum().sum() == 0
    assert cleaned[config.ZERO_FILL_COLS].isna().sum().sum() == 0


def test_clean_data_marks_absent_features_as_none_string(raw):
    cleaned = clean_data(raw)
    no_garage = raw["Garage Type"].isna()
    assert (cleaned.loc[no_garage, "Garage Type"] == "None").all()


def test_clean_data_converts_codes_to_strings(raw):
    cleaned = clean_data(raw)
    for col in config.CATEGORICAL_AS_STR:
        assert cleaned[col].dtype.kind not in "iuf", f"{col} still numeric"


def test_clean_data_does_not_modify_input(raw):
    before = raw.copy()
    clean_data(raw)
    assert raw.equals(before)


def test_clean_data_works_without_target(raw):
    cleaned = clean_data(raw.drop(columns=config.TARGET))
    assert len(cleaned) == len(raw)


def test_remove_outliers_drops_only_the_three_partial_sales(raw):
    result = remove_outliers(clean_data(raw))
    assert len(result) == len(raw) - 3
    big = result[result["Gr Liv Area"] > 4000]
    assert (big["SalePrice"] >= 300000).all()
