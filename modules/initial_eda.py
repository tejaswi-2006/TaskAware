"""
Initial exploratory data analysis for TaskAware.

Uses the central schema analyzer so that column-role
information is consistent across the application.
"""

import numpy as np
import pandas as pd

from modules.schema_analyzer import analyze_schema


def perform_initial_eda(df, target_column=None):
    """
    Perform initial EDA on the uploaded dataset.

    Returns
    -------
    dict
        - shape
        - column roles
        - unique values
        - missing values
        - numerical summary
        - categorical summary
        - text summary
        - date summary
        - unsupported columns
        - id candidates
        - duplicate rows
    """

    schema = analyze_schema(
        df,
        target_column=target_column,
    )

    numerical_columns = schema["numerical_columns"]
    categorical_columns = schema["categorical_columns"]
    boolean_columns = schema["boolean_columns"]
    text_columns = schema["text_columns"]
    date_columns = schema["date_columns"]

    unique_values = {}

    for column in df.columns:
        try:
            unique_values[str(column)] = int(
                df[column].nunique(dropna=True)
            )
        except Exception:
            unique_values[str(column)] = None

    # ------------------------------------------------
    # NUMERICAL SUMMARY
    # ------------------------------------------------

    if numerical_columns:
        statistical_summary = (
            df[numerical_columns]
            .describe()
            .round(2)
            .to_dict()
        )
    else:
        statistical_summary = {}

    # ------------------------------------------------
    # CATEGORICAL SUMMARY
    # ------------------------------------------------

    categorical_summary = {}

    for column in categorical_columns + boolean_columns:

        series = df[column].dropna()

        if len(series) == 0:
            continue

        counts = series.value_counts()

        categorical_summary[str(column)] = {
            "unique_count": int(series.nunique()),
            "most_common": str(counts.index[0]),
            "most_common_count": int(counts.iloc[0]),
            "most_common_ratio": round(
                float(counts.iloc[0] / len(series)),
                3,
            ),
        }

    # ------------------------------------------------
    # TEXT SUMMARY
    # ------------------------------------------------

    text_summary = {}

    for column in text_columns:

        series = df[column].astype(str)

        lengths = series.str.len()

        word_counts = series.str.split().str.len()

        text_summary[str(column)] = {
            "missing": int(df[column].isnull().sum()),
            "unique_count": int(
                series.nunique()
            ),
            "avg_length": round(
                float(lengths.mean()),
                2,
            ),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "avg_words": round(
                float(word_counts.mean()),
                2,
            ),
            "vocabulary_estimate": min(
                int(
                    series.str.split()
                    .explode()
                    .nunique()
                ),
                100000,
            ),
        }

    # ------------------------------------------------
    # DATE SUMMARY
    # ------------------------------------------------

    date_summary = {}

    for column in date_columns:

        parsed = pd.to_datetime(
            df[column],
            errors="coerce",
            format="mixed",
        )

        valid = parsed.dropna()

        date_summary[str(column)] = {
            "missing": int(parsed.isnull().sum()),
            "min": str(valid.min()) if len(valid) else None,
            "max": str(valid.max()) if len(valid) else None,
            "range_days": (
                int((valid.max() - valid.min()).days)
                if len(valid) >= 2
                else None
            ),
        }

    eda_results = {
        "shape": list(df.shape),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "column_roles": {
            record["column"]: record["role"]
            for record in schema["columns"]
        },
        "roles": schema["by_role"],
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "boolean_columns": boolean_columns,
        "text_columns": text_columns,
        "date_columns": date_columns,
        "id_columns": schema["id_columns"],
        "unsupported_columns": schema["unsupported_columns"],
        "constant_columns": schema["constant_columns"],
        "unique_values": unique_values,
        "statistical_summary": statistical_summary,
        "categorical_summary": categorical_summary,
        "text_summary": text_summary,
        "date_summary": date_summary,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "schema": schema,
    }

    return eda_results