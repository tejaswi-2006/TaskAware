import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.preprocessing import (
    build_preprocessing_pipeline,
    identify_column_roles
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_model_feature_importance(
    pipeline,
    X,
    y,
    problem_type
):
    """
    Extract feature importance from tree-based models
    when available. Returns a list of importance dicts.
    """

    model = pipeline.named_steps.get("model")

    if model is None:
        return []

    if not hasattr(model, "feature_importances_"):
        return []

    try:

        # Transform X with the fitted preprocessor so the
        # feature names align with the model's view.
        preprocessor = pipeline.named_steps.get(
            "preprocessor"
        )

        feature_names = _get_feature_names(
            pipeline,
            X
        )

        importances = model.feature_importances_

        if len(importances) != len(feature_names):
            importances = importances[:len(feature_names)]

        items = []

        for name, importance in zip(
            feature_names,
            importances
        ):

            items.append({
                "feature": name,
                "importance": round(
                    float(importance),
                    4
                )
            })

        items.sort(
            key=lambda item: item["importance"],
            reverse=True
        )

        return items

    except Exception:
        return []


# ============================================================
# SHAP ANALYSIS
# ============================================================

def compute_shap_values(
    pipeline,
    X,
    problem_type
):
    """
    Compute SHAP values for the model.

    Uses the full pipeline so that input data flows through
    preprocessing exactly as during training.

    Returns (shap_values, feature_names, model).
    If SHAP cannot be applied, returns (None, ...).
    """

    import shap

    model = pipeline.named_steps.get("model")

    if model is None:
        return None, [], None

    preprocessor = pipeline.named_steps.get(
        "preprocessor"
    )

    try:

        X_transformed = preprocessor.transform(X)

        feature_names = _get_feature_names(
            pipeline,
            X
        )

        X_df = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )

        # Choose the explainer based on model type
        is_tree = isinstance(
            model,
            (
                __import__(
                    "sklearn.ensemble",
                    fromlist=["*"]
                ).RandomForestClassifier,
                __import__(
                    "sklearn.ensemble",
                    fromlist=["*"]
                ).RandomForestRegressor,
                __import__(
                    "sklearn.tree",
                    fromlist=["*"]
                ).DecisionTreeClassifier,
                __import__(
                    "sklearn.tree",
                    fromlist=["*"]
                ).DecisionTreeRegressor
            )
        )

        if is_tree:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_df)

        # Use a sample for speed
        sample_size = min(
            len(X_df),
            200
        )

        X_sample = X_df.sample(
            n=sample_size,
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            shap_values = explainer.shap_values(X_sample)

        return shap_values, X_sample, explainer

    except Exception:
        return None, [], None


def _get_feature_names(
    pipeline,
    X
):
    """
    Retrieve feature names after preprocessing.
    """

    try:

        preprocessor = pipeline.named_steps.get(
            "preprocessor"
        )

        return list(
            preprocessor.get_feature_names_out()
        )

    except Exception:

        return list(X.columns)


# ============================================================
# SHAP SUMMARIZATION
# ============================================================

def summarize_shap(
    shap_values,
    feature_names,
    problem_type
):
    """
    Convert raw SHAP values into a readable per-feature
    summary.

    Returns
    -------
    list of dicts
        sorted by mean absolute SHAP value.
    """

    try:

        values = np.asarray(
            shap_values,
            dtype=float
        )

        if values.ndim == 3:
            # For multiclass: average absolute contribution
            # across classes
            values = values.mean(axis=2)

        if values.ndim != 2:
            return []

        mean_abs = np.abs(values).mean(axis=0)

        items = []

        for i, name in enumerate(feature_names):

            if i >= len(mean_abs):
                break

            items.append({
                "feature": name,
                "mean_abs_shap": round(
                    float(mean_abs[i]),
                    4
                )
            })

        items.sort(
            key=lambda item: item["mean_abs_shap"],
            reverse=True
        )

        return items

    except Exception:
        return []


# ============================================================
# SHAP BAR CHART
# ============================================================

def generate_shap_chart(
    shap_values,
    X_sample,
    problem_type,
    output_path
):
    """
    Generate a SHAP summary bar chart.
    """

    import shap

    if shap_values is None:
        return False

    try:

        plt.figure()

        shap.summary_plot(
            shap_values,
            X_sample,
            show=False,
            max_display=15
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=100,
            bbox_inches="tight"
        )

        plt.close()

        return True

    except Exception:
        return False


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def perform_explainability(
    df,
    target_column,
    problem_type,
    pipeline=None,
    X=None,
    y=None,
    charts_dir=None,
    chart_id=None
):
    """
    Explain the best model using SHAP.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataset (used to rebuild if needed).
    target_column : str
        Target column.
    problem_type : str
        Classification or Regression.
    pipeline : sklearn Pipeline or None
        Fitted pipeline (preprocessor + model).
    X, y : data or None
        Feature matrix and target.
    charts_dir, chart_id : str or None
        Chart output settings.

    Returns
    -------
    dict
        SHAP summary, importance, and chart info.
    """

    result = {
        "success": False,
        "message": "",
        "shap_summary": [],
        "feature_importance": [],
        "chart_file": None,
        "top_features": []
    }

    if pipeline is None:
        result["message"] = "No trained model was provided."
        return result

    # --------------------------------------------------------
    # SHAP VALUES
    # --------------------------------------------------------

    shap_values, X_sample, explainer = compute_shap_values(
        pipeline,
        X,
        problem_type
    )

    feature_names = (
        list(X_sample.columns)
        if X_sample is not None
        else []
    )

    if shap_values is not None:

        shap_summary = summarize_shap(
            shap_values,
            feature_names,
            problem_type
        )

        result["shap_summary"] = shap_summary

        result["top_features"] = [
            item["feature"]
            for item in shap_summary[:5]
        ]

        # Chart
        if charts_dir:

            os.makedirs(
                charts_dir,
                exist_ok=True
            )

            base = chart_id or "shap"

            chart_path = os.path.join(
                charts_dir,
                f"{base}_shap.png"
            )

            if generate_shap_chart(
                shap_values,
                X_sample,
                problem_type,
                chart_path
            ):

                result["chart_file"] = os.path.basename(
                    chart_path
                )

        result["success"] = True

        result["message"] = (
            "SHAP analysis completed successfully."
        )

    # --------------------------------------------------------
    # MODEL FEATURE IMPORTANCE (fallback / complement)
    # --------------------------------------------------------

    if X is not None and y is not None:

        importance = get_model_feature_importance(
            pipeline,
            X,
            y,
            problem_type
        )

        if importance:
            result["feature_importance"] = importance

            if not result["shap_summary"]:
                result["top_features"] = [
                    item["feature"]
                    for item in importance[:5]
                ]

    if not result["shap_summary"] and not result["feature_importance"]:

        result["message"] = (
            "Explainability could not be computed for "
            "this model."
        )

    return result