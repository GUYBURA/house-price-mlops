import pytest

from src import config
from src.validate import DataValidationError, find_problems, validate


def has_problem(problems, text):
    return any(text in p for p in problems)


def test_real_data_passes(raw):
    """Most important test: a validator that rejects good data is useless."""
    assert find_problems(raw) == []


def test_prediction_mode_without_target_passes(raw):
    one_house = raw.drop(columns=[config.TARGET, "Order", "PID"]).head(1)
    assert find_problems(one_house, require_target=False) == []


def test_missing_column_is_reported_by_name(raw):
    problems = find_problems(raw.drop(columns="Year Built"))
    assert problems == ["missing columns: ['Year Built']"]


def test_training_requires_target(raw):
    problems = find_problems(raw.drop(columns=config.TARGET))
    assert has_problem(problems, config.TARGET)


def test_too_few_rows_for_training(raw):
    problems = find_problems(raw.head(10))
    assert has_problem(problems, "only 10 rows")


def test_non_positive_price(raw):
    raw.loc[:2, config.TARGET] = 0
    assert has_problem(find_problems(raw), "SalePrice: 3 rows outside")


def test_missing_price(raw):
    raw[config.TARGET] = raw[config.TARGET].astype(float)
    raw.loc[0, config.TARGET] = None
    assert has_problem(find_problems(raw), "SalePrice: 1 missing values")


def test_living_area_all_zero(raw):
    """The 'upstream bug' case: column arrives as all zeros."""
    raw["Gr Liv Area"] = 0
    problems = find_problems(raw)
    assert has_problem(problems, "Gr Liv Area: every value is 0")
    assert has_problem(problems, "Gr Liv Area: 2930 rows outside")


def test_text_in_numeric_column(raw):
    raw["Gr Liv Area"] = raw["Gr Liv Area"].astype(str)
    assert has_problem(find_problems(raw), "Gr Liv Area: expected numbers")


def test_overall_qual_out_of_scale(raw):
    raw.loc[0, "Overall Qual"] = 11
    assert has_problem(find_problems(raw), "Overall Qual: 1 rows outside 1-10")


def test_built_in_the_future(raw):
    raw.loc[0, "Year Built"] = 2999
    problems = find_problems(raw)
    assert has_problem(problems, "Year Built: 1 rows outside")


def test_sold_long_before_built(raw):
    raw.loc[0, "Year Built"] = raw.loc[0, "Yr Sold"] + 5
    assert has_problem(find_problems(raw), "Yr Sold: 1 rows")


def test_unknown_quality_grade(raw):
    raw.loc[0, "Kitchen Qual"] = "Excellent"
    assert has_problem(find_problems(raw), "Kitchen Qual: unknown grades ['Excellent']")


def test_missing_quality_grade_is_allowed(raw):
    """NaN means 'no basement' and is handled by clean_data, not an error."""
    raw.loc[0, "Bsmt Qual"] = None
    assert find_problems(raw) == []


def test_all_problems_reported_at_once(raw):
    raw.loc[0, "Overall Qual"] = 0
    raw.loc[0, "Kitchen Qual"] = "Excellent"
    assert len(find_problems(raw)) == 2


def test_validate_raises_with_every_problem_listed(raw):
    raw.loc[0, "Overall Qual"] = 0
    raw.loc[0, "Kitchen Qual"] = "Excellent"
    with pytest.raises(DataValidationError) as error:
        validate(raw)
    assert "Overall Qual" in str(error.value)
    assert "Kitchen Qual" in str(error.value)


def test_validate_passes_silently_on_good_data(raw):
    validate(raw)
