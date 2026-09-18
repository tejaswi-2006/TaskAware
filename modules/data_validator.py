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

    validation_results = {
        "is_empty": df.empty,
        "rows": rows,
        "columns": cols,
        "duplicate_rows": int(df.duplicated().sum()),
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