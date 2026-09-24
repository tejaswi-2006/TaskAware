"""
Problem type detection for TaskAware.

Decides whether the selected target represents a
classification or regression problem.

Uses the central schema analyzer so that text, date and
unsupported targets are reported clearly instead of being
silently misclassified.
"""

import pandas as pd

from modules.schema_analyzer import analyze_schema


def detect_problem_type(
    df,
    target_column,
    objective=None,
    schema=None,
):
    """
    Detect whether the selected target represents
    classification or regression.

    Returns
    -------
    dict
        success, problem_type, target_column, data_type,
        unique_values, message, warning (optional).
    """

    if schema is None:
        schema = analyze_schema(
            df,
            target_column=target_column,
        )

    # ========================================================
    # CHECK TARGET
    # ========================================================

    if target_column not in df.columns:

        return {
            "success": False,
            "problem_type": None,
            "message":
                "Selected target column does not exist.",
        }

    target_record = next(
        (
            record for record in schema["columns"]
            if record["column"] == target_column
        ),
        None,
    )

    target_role = (
        target_record["role"]
        if target_record
        else None
    )

    storage_dtype = (
        target_record.get("raw_dtype", target_record.get("dtype"))
        if target_record
        else str(df[target_column].dtype)
    )

    semantic_type = (
        target_record.get("semantic_type")
        if target_record
        else None
    )

    target = df[target_column].dropna()

    # ========================================================
    # EMPTY TARGET
    # ========================================================

    if target.empty:

        return {
            "success": False,
            "problem_type": None,
            "target_column": target_column,
            "message":
                "Target column contains no valid values.",
        }

    unique_count = int(target.nunique())

    total_count = len(target)

    unique_ratio = (
        unique_count / total_count
    )

    # ========================================================
    # UNSUPPORTED / ID / CONSTANT TARGET
    # ========================================================

    if target_role in (
        "identifier",
        "unsupported",
        "constant",
    ):

        return {
            "success": False,
            "problem_type": None,
            "target_column": target_column,
            "data_type": target_role,
            "unique_values": unique_count,
            "message": (
                f"'{target_column}' cannot be used as a "
                f"target because it was detected as an "
                f"{target_role} column."
            ),
        }

    # ========================================================
    # TEXT TARGET
    # ========================================================

    if target_role == "text":

        return {
            "success": False,
            "problem_type": None,
            "target_column": target_column,
            "data_type": "text",
            "unique_values": unique_count,
            "message": (
                "'{target_column}' is a free-text column. "
                "The current prediction engine supports "
                "structured categorical or numerical "
                "targets, not free-text prediction."
            ),
            "warning": True,
        }

    # ========================================================
    # DATETIME TARGET
    # ========================================================

    if target_role == "datetime":

        return {
            "success": False,
            "problem_type": None,
            "target_column": target_column,
            "data_type": "datetime",
            "unique_values": unique_count,
            "message": (
                "'{target_column}' is a date column. "
                "TaskAware does not predict raw dates."
            ),
            "warning": True,
        }

    # ========================================================
    # DATA TYPE
    # ========================================================

    if target_role == "numeric":
        data_type = "numerical"
    elif target_role == "boolean":
        data_type = "categorical"
    else:
        data_type = (
            "categorical"
            if not pd.api.types.is_numeric_dtype(target)
            else "numerical"
        )

    # ========================================================
    # OBJECTIVE HINT
    # ========================================================

    objective_text = (
        str(objective).lower().strip()
        if objective
        else ""
    )

    regression_words = [
        "regression",
        "predict score",
        "predict price",
        "predict salary",
        "predict income",
        "predict sales",
        "predict amount",
        "predict value",
    ]

    classification_words = [
        "classify",
        "classification",
        "categorize",
        "categorise",
        "category",
        "class",
        "label",
    ]

    objective_requests_regression = any(
        word in objective_text
        for word in regression_words
    )

    objective_requests_classification = any(
        word in objective_text
        for word in classification_words
    )

    # ========================================================
    # CATEGORICAL / BOOLEAN TARGET
    # ========================================================

    if target_role in ("categorical", "boolean"):

        return {
            "success": True,
            "problem_type": "Classification",
            "target_column": target_column,
            "data_type": data_type,
            "storage_dtype": storage_dtype,
            "semantic_type": semantic_type,
            "unique_values": unique_count,
            "message": (
                "The selected target is categorical, "
                "so TaskAware treats the problem as "
                "classification."
            ),
        }

    # ========================================================
    # NUMERICAL TARGET
    # ========================================================

    if target_role == "numeric":

        if (
            unique_count <= 10
            and unique_ratio < 0.05
        ):

            if objective_requests_regression:

                return {
                    "success": True,
                    "problem_type": "Regression",
                    "target_column": target_column,
                    "data_type": data_type,
                    "unique_values": unique_count,
                    "message": (
                        "The target is numerical with a "
                        "small number of distinct values. "
                        "The stated objective indicates "
                        "that numerical prediction is "
                        "intended, so TaskAware treats it "
                        "as regression."
                    ),
                }

            return {
                "success": True,
                "problem_type": "Classification",
                "target_column": target_column,
                "data_type": data_type,
                "unique_values": unique_count,
                "message": (
                    "The numerical target contains a small "
                    "number of distinct values and may "
                    "represent class labels. TaskAware "
                    "treats it as classification unless the "
                    "objective indicates a numerical "
                    "prediction."
                ),
            }

        return {
            "success": True,
            "problem_type": "Regression",
            "target_column": target_column,
            "data_type": data_type,
            "storage_dtype": storage_dtype,
            "semantic_type": semantic_type,
            "unique_values": unique_count,
            "message": (
                "The selected target contains numerical "
                "values with sufficient variation, so "
                "TaskAware treats the problem as regression."
            ),
        }

    # ========================================================
    # UNKNOWN
    # ========================================================

    return {
        "success": False,
        "problem_type": None,
        "target_column": target_column,
        "data_type": data_type,
        "unique_values": unique_count,
        "message": "Unable to determine the problem type.",
    }