from datetime import date

import pandas as pd

from src import config


class DataValidationError(ValueError):
    """Raised when incoming data is broken."""


def find_problems(df: pd.DataFrame, require_target: bool = True) -> list[str]:
    """Check raw data before cleaning. Returns every problem found, empty list if valid."""
    problems = []

    required = config.REQUIRED_COLS + ([config.TARGET] if require_target else [])
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        # later checks would crash on the missing columns, so stop here
        problems.append(f"missing columns: {missing_columns}")
        return problems

    if require_target:
        if len(df) < config.MIN_TRAIN_ROWS:
            problems.append(f"only {len(df)} rows, need at least {config.MIN_TRAIN_ROWS} to train")
        problems += _check_numeric(df, config.TARGET, low=1, high=None)

    for col, (low, high) in config.VALUE_RANGES.items():
        problems += _check_numeric(df, col, low, high)

    next_year = date.today().year + 1
    problems += _check_numeric(df, "Year Built", low=1800, high=next_year)
    # Ames has one house sold a year before its recorded build year, so allow -1
    sold_before_built = (df["Yr Sold"] < df["Year Built"] - 1).sum()
    if sold_before_built:
        problems.append(f"Yr Sold: {sold_before_built} rows sold more than a year before Year Built")

    for col in config.QUAL_COLS:
        # missing means "feature absent" (clean_data fills "None"), so only check present values
        unknown = set(df[col].dropna().unique()) - config.VALID_QUAL_GRADES
        if unknown:
            problems.append(f"{col}: unknown grades {sorted(unknown)}")

    for col in config.NON_ZERO_COLS + ([config.TARGET] if require_target else []):
        if pd.api.types.is_numeric_dtype(df[col]) and len(df) and (df[col] == 0).all():
            problems.append(f"{col}: every value is 0")

    return problems


def _check_numeric(df: pd.DataFrame, col: str, low: float, high: float | None) -> list[str]:
    """Column must be numeric, have no missing values, and stay within [low, high]."""
    values = df[col]
    if not pd.api.types.is_numeric_dtype(values):
        return [f"{col}: expected numbers, got {values.dtype}"]

    problems = []
    n_missing = values.isna().sum()
    if n_missing:
        problems.append(f"{col}: {n_missing} missing values")

    present = values.dropna()
    out_of_range = (present < low) | (present > high) if high is not None else present < low
    if out_of_range.sum():
        bounds = f"{low}-{high}" if high is not None else f">= {low}"
        problems.append(f"{col}: {out_of_range.sum()} rows outside {bounds}")
    return problems


def validate(df: pd.DataFrame, require_target: bool = True) -> None:
    problems = find_problems(df, require_target)
    if problems:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(problems))
