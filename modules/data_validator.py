"""
Dataset validation for TaskAware.

Uses the central schema analyzer so that identifier, text,
date and unsupported detection is consistent across the
whole application.
"""

from modules.schema_analyzer import analyze_schema


def validate_dataset(df, target_column=None):
    """
    Perform comprehensive validation on the uploaded dataset.

    Returns
    -------
    dict
        - is_empty
        - rows
        - columns
        - duplicate_rows
        - missing_values (per column)
        - total_missing
        - duplicate_column_names
        - constant_columns
        - possible_id_columns (with reasons)
        - text_columns
        - date_columns
        - unsupported_columns
        - data_types
        - roles (per column)
        - schema (full central schema)
    """

    schema = analyze_schema(
        df,
        target_column=target_column,
    )

    rows, cols = df.shape

    duplicate_column_names = [
        str(name)
        for name, count
        in df.columns.value_counts().items()
        if count > 1
    ]

    id_reasons = {}

    for record in schema["columns"]:
        if record["role"] == "identifier":
            id_reasons[record["column"]] = record["reasons"]

    text_stats = {}

    for record in schema["columns"]:
        if record["role"] == "text":
            text_stats[record["column"]] = record["shape"]

    date_stats = {}

    for column in schema["date_columns"]:
        series = pd_to_datetime_safe(df[column])
        if series is not None and len(series) > 0:
            date_stats[column] = {
                "min": str(series.min()),
                "max": str(series.max()),
                "missing": int(df[column].isnull().sum()),
            }

    duplicate_check = safe_duplicate_check(df)

    validation_results = {
        "is_empty": df.empty,
        "rows": rows,
        "columns": cols,
        "duplicate_rows": duplicate_check[
            "duplicate_rows_detected"
        ],
        "duplicate_check_status": duplicate_check[
            "duplicate_check_status"
        ],
        "duplicate_check_scope": duplicate_check[
            "duplicate_check_scope"
        ],
        "duplicate_check_warning": duplicate_check[
            "duplicate_check_warning"
        ],
        "unsupported_columns_ignored_for_duplicate_check": (
            duplicate_check[
                "unsupported_columns_ignored_for_duplicate_check"
            ]
        ),
        "duplicate_rows_detected": duplicate_check[
            "duplicate_rows_detected"
        ],
        "duplicate_check_complete": duplicate_check[
            "duplicate_check_complete"
        ],
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_column_names": duplicate_column_names,
        "constant_columns": schema["constant_columns"],
        "possible_id_columns": schema["id_columns"],
        "possible_id_reasons": id_reasons,
        "text_columns": schema["text_columns"],
        "text_stats": text_stats,
        "date_columns": schema["date_columns"],
        "date_stats": date_stats,
        "unsupported_columns": schema["unsupported_columns"],
        "data_types": (
            df.dtypes.astype(str).to_dict()
        ),
        "roles": {
            record["column"]: record["role"]
            for record in schema["columns"]
        },
        "by_role": schema["by_role"],
        "schema": schema,
    }

    return validation_results


def find_unhashable_columns(df):
    """
    Return columns whose values cannot be hashed (dicts,
    lists, sets, nested JSON, mixed unhashable objects).

    Each column is probed individually so that one nested
    column does not hide the status of the others.
    """
    bad = []

    for column in df.columns:
        series = df[column]

        # Numeric, boolean and datetime dtypes are always
        # hashable; only object columns can hide nested
        # values. Probing with hash() directly is
        # deterministic, unlike Series.duplicated(),
        # which silently tolerates some unhashables.
        if series.dtype != object:
            continue

        try:
            series.map(hash)
        except Exception:
            bad.append(column)

    return bad


def safe_duplicate_check(df, exclude_columns=None):
    """
    Duplicate-row check that never crashes on nested or
    unhashable values.

    Strategy
    --------
    - Run the normal check when all values are hashable.
    - Otherwise exclude only the unhashable columns from
      duplicate hashing and mark the check as partial.
    - The original data is never modified.

    Returns metadata with duplicate_check_status
    (complete/partial/skipped), scope, warning, ignored
    columns, detected count and completeness flag.
    """
    excluded = list(exclude_columns or [])

    auto = find_unhashable_columns(df)

    for column in auto:
        if column not in excluded:
            excluded.append(column)

    usable = [
        column
        for column in df.columns
        if column not in excluded
    ]

    if len(df.columns) == 0 or len(df) == 0:
        return {
            "duplicate_rows_detected": 0,
            "duplicate_check_status": "complete",
            "duplicate_check_scope": "all columns",
            "duplicate_check_warning": None,
            "unsupported_columns_ignored_for_duplicate_check": [],
            "duplicate_check_complete": True,
        }

    if not usable:
        return {
            "duplicate_rows_detected": 0,
            "duplicate_check_status": "skipped",
            "duplicate_check_scope": (
                "no hashable columns available"
            ),
            "duplicate_check_warning": (
                "Duplicate check could not run because "
                "every column contains nested or unhashable "
                "values."
            ),
            "unsupported_columns_ignored_for_duplicate_check": (
                [str(c) for c in excluded]
            ),
            "duplicate_check_complete": False,
        }

    try:
        detected = int(
            df[usable].duplicated().sum()
        )
    except Exception as exc:
        return {
            "duplicate_rows_detected": 0,
            "duplicate_check_status": "skipped",
            "duplicate_check_scope": (
                "duplicate hashing failed"
            ),
            "duplicate_check_warning": (
                "Duplicate check could not run safely: "
                f"{str(exc)[:160]}"
            ),
            "unsupported_columns_ignored_for_duplicate_check": (
                [str(c) for c in excluded]
            ),
            "duplicate_check_complete": False,
        }

    if excluded:
        names = ", ".join(str(c) for c in excluded)

        return {
            "duplicate_rows_detected": detected,
            "duplicate_check_status": "partial",
            "duplicate_check_scope": (
                f"all columns except {names}"
            ),
            "duplicate_check_warning": (
                "Partial duplicate check completed. Nested "
                f"object columns ({names}) were excluded "
                "from duplicate hashing. The duplicate "
                "count may not include duplicates "
                "distinguishable only through nested object "
                "content."
            ),
            "unsupported_columns_ignored_for_duplicate_check": (
                [str(c) for c in excluded]
            ),
            "duplicate_check_complete": False,
        }

    return {
        "duplicate_rows_detected": detected,
        "duplicate_check_status": "complete",
        "duplicate_check_scope": "all columns",
        "duplicate_check_warning": None,
        "unsupported_columns_ignored_for_duplicate_check": [],
        "duplicate_check_complete": True,
    }


def pd_to_datetime_safe(series):
    """
    Best-effort conversion of a series to datetime.
    Returns None when the column is not date-like.
    """
    import pandas as pd

    try:
        return pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )
    except Exception:
        return None