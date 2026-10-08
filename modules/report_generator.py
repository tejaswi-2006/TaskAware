import os
from datetime import datetime

import pandas as pd


# ============================================================
# REPORT DATA COLLECTION
# ============================================================

def collect_report_data(
    df,
    filename,
    objective,
    target_column,
    problem_type,
    explanation,
    selected_model=None
):
    """
    Collect all structured information needed for the
    final report.

    This function recomputes the pipeline results so the
    report is self-contained and reflects the current
    dataset.
    """

    from modules.data_validator import validate_dataset
    from modules.detailed_eda import perform_detailed_eda
    from modules.preprocessing import prepare_preprocessing
    from modules.schema_analyzer import analyze_schema

    rows, columns = df.shape

    schema = analyze_schema(
        df,
        target_column=(
            target_column
            if problem_type != "Clustering"
            else None
        ),
    )

    from modules.data_validator import safe_duplicate_check

    duplicate_check = safe_duplicate_check(df)

    data_quality = {
        "rows": rows,
        "columns": columns,
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
        "missing_cells": int(
            df.isnull().sum().sum()
        ),
        "missing_ratio": round(
            float(
                df.isnull().sum().sum() /
                (rows * columns)
            ),
            4
        )
    }

    validation = validate_dataset(
        df,
        target_column=(
            target_column
            if problem_type != "Clustering"
            else None
        ),
    )

    data_quality["constant_columns"] = (
        validation.get("constant_columns", [])
    )

    data_quality["possible_id_columns"] = (
        validation.get("possible_id_columns", [])
    )

    data_quality["unsupported_columns"] = (
        validation.get("unsupported_columns", [])
    )

    data_quality["text_columns"] = (
        validation.get("text_columns", [])
    )

    data_quality["date_columns"] = (
        validation.get("date_columns", [])
    )

    # ------------------------------------------------
    # DETAILED EDA
    # ------------------------------------------------

    eda = perform_detailed_eda(
        df,
        target_column=(
            target_column
            if problem_type != "Clustering"
            else None
        ),
        problem_type=problem_type
    )

    # ------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------

    prep = prepare_preprocessing(
        df,
        target_column,
        problem_type
    )

    # ------------------------------------------------
    # MODEL RESULTS
    # ------------------------------------------------

    model_results = None

    used_chronological_split = False

    used_group_split = None

    if problem_type == "Clustering":

        from modules.clustering import perform_clustering

        cluster_result = perform_clustering(
            df,
            objective=(
                objective
                if objective
                else None
            )
        )

        model_results = {
            "problem_type": "Clustering",
            "success": cluster_result.get("success"),
            "n_clusters": cluster_result.get("n_clusters"),
            "silhouette": (
                cluster_result.get("best", {}).get(
                    "silhouette"
                )
                if cluster_result.get("best")
                else None
            ),
            "davies_bouldin": (
                cluster_result.get("best", {}).get(
                    "davies_bouldin"
                )
                if cluster_result.get("best")
                else None
            ),
            "stability": cluster_result.get("stability"),
            "profiles": cluster_result.get("profiles"),
            "evaluations": cluster_result.get(
                "evaluations"
            ),
            "representation_note": (
                cluster_result.get("representation_note")
            ),
        }

    else:

        from modules.model_trainer import train_models
        from modules.model_selector import select_best_model

        training = train_models(
            df,
            target_column,
            problem_type
        )

        best = select_best_model(
            training["results"],
            problem_type
        )

        used_chronological_split = (
            training.get("used_chronological_split", False)
        )

        used_group_split = training.get("used_group_split")

        # Actual explanation method from the explanation
        # layer (never assumed to be SHAP).
        explanation_method = None
        explanation_top_features = []

        try:

            from modules.model_trainer import train_named_model
            from modules.explainability import (
                perform_explainability,
            )

            if best is not None:
                fitted = train_named_model(
                    df,
                    target_column,
                    problem_type,
                    best["name"],
                )

                explain_result = perform_explainability(
                    df,
                    target_column,
                    problem_type,
                    pipeline=fitted["pipeline"],
                    X=fitted["X_test"],
                    y=fitted["y_test"],
                )

                explanation_method = explain_result.get(
                    "method"
                )
                explanation_top_features = (
                    explain_result.get("top_features", [])[:5]
                )

        except Exception:
            pass

        model_results = {
            "problem_type": problem_type,
            "target_column": target_column,
            "best_model": best,
            "results": training["results"],
            "n_classes": training.get("n_classes"),
            "X_train_shape": training["X_train_shape"],
            "X_test_shape": training["X_test_shape"],
            "used_chronological_split": (
                used_chronological_split
            ),
            "used_group_split": used_group_split,
            "n_transformed_features": training.get(
                "n_transformed_features"
            ),
            "feature_accounting": training.get(
                "feature_accounting"
            ),
            "explanation_method": explanation_method,
            "explanation_top_features": (
                explanation_top_features
            ),
            "row_info": training.get("row_info"),
        }

        if selected_model and not best:
            model_results["best_model"] = {
                "name": selected_model
            }

    # ------------------------------------------------
    # WARNINGS
    # ------------------------------------------------

    warnings = []

    if schema["is_time_series"] and schema["time_column"]:

        warnings.append(
            f"The dataset contains a date column "
            f"('{schema['time_column']}'). "
            "TaskAware applied a chronological split when "
            "training so the model never sees future data. "
            "Forecasting support is limited; results are "
            "best treated as time-aware validation rather "
            "than full production forecasting."
        )

    if schema["unsupported_columns"]:

        warnings.append(
            f"The following columns contain unsupported "
            "types and were excluded from the analysis: "
            f"{', '.join(schema['unsupported_columns'])}"
        )

    if schema["constant_columns"]:

        warnings.append(
            f"The following constant columns carried no "
            "information and were excluded: "
            f"{', '.join(schema['constant_columns'])}"
        )

    # ------------------------------------------------
    # COLUMN ROLES
    # ------------------------------------------------

    column_roles = [
        {
            "column": record["column"],
            "role": record["role"],
            "dtype": record["dtype"],
            "unique_count": record["unique_count"],
            "missing_count": record["missing_count"],
            "reasons": record["reasons"],
            "is_target": record["is_target"],
        }
        for record in schema["columns"]
    ]

    # ------------------------------------------------
    # MODELING ROWS (missing-target exclusion)
    # ------------------------------------------------

    _model_row_info = (
        (model_results or {}).get("row_info") or {}
    )

    data_quality["rows_available_for_modeling"] = (
        _model_row_info.get("rows_available_for_modeling", rows)
    )
    data_quality["rows_excluded_missing_target"] = (
        _model_row_info.get("rows_excluded_missing_target", 0)
    )
    data_quality["target_warning"] = _model_row_info.get(
        "target_warning"
    )

    # ------------------------------------------------
    # REPORT
    # ------------------------------------------------

    return {
        "filename": filename,
        "objective": objective or None,
        "target_column": (
            target_column
            if problem_type != "Clustering"
            else None
        ),
        "problem_type": problem_type,
        "selected_model": selected_model,
        "data_quality": data_quality,
        "validation": validation,
        "eda": eda,
        "prep": prep,
        "model_results": model_results,
        "explanation": explanation,
        "schema": schema,
        "group_validation": {
            "group_column": (
                (schema.get("group_details") or [{}])[0].get(
                    "column"
                )
                if schema.get("group_details")
                else None
            ),
            "strategy": schema.get(
                "validation_strategy", {}
            ).get("strategy"),
            "reason": schema.get(
                "validation_strategy", {}
            ).get("reason"),
            "confidence": (
                (schema.get("group_details") or [{}])[0].get(
                    "confidence"
                )
                if schema.get("group_details")
                else None
            ),
            "details": schema.get("group_details", []),
        },
        "column_roles": column_roles,
        "warnings": warnings,
        "random_seed": 42,
        "software": {
            "module": "TaskAware",
            "version": "1.0.0",
        },
        "generated_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "columns": list(df.columns),
        "n_rows": rows,
        "n_columns": columns
    }


# ============================================================
# HTML REPORT GENERATION
# ============================================================

def generate_html_report(report_data):
    """
    Build a complete HTML report string from the
    structured report data.
    """

    from flask import render_template

    # Render the full report template with the collected data
    return render_template(
        "report.html",
        report=report_data
    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    report_data,
    reports_dir,
    report_id=None
):
    """
    Save the rendered report to disk and return the path.
    """

    os.makedirs(
        reports_dir,
        exist_ok=True
    )

    base = os.path.splitext(
        report_data["filename"]
    )[0]

    if report_id:
        base = f"{base}_{report_id}"

    base = base.replace(" ", "_")

    html_path = os.path.join(
        reports_dir,
        f"{base}_report.html"
    )

    content = generate_html_report(
        report_data
    )

    with open(
        html_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(content)

    return html_path, content