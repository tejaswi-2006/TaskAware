"""
Explainability for TaskAware.

Explains supervised models using SHAP where compatible, and
falls back to model coefficients or permutation importance
otherwise.

Supports:
    - dense numerical features
    - one-hot encoded features
    - sparse TF-IDF features
    - mixed transformed matrices

Term names keep their source column prefix (e.g.
"review:excellent"). Displayed features are limited so huge
text vocabularies never produce oversized output.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from modules.schema_analyzer import analyze_schema


# ============================================================
# FEATURE NAME RECOVERY
# ============================================================

def _get_feature_names(pipeline, X):
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


def _source_column(name, text_columns):
    """
    Recover the source column of a transformed feature name.
    """
    if ":" in name:
        source = name.split(":", 1)[0]
        if source in text_columns:
            return source
    return None


# ============================================================
# MODEL FEATURE IMPORTANCE (trees)
# ============================================================

def get_model_feature_importance(
    pipeline,
    X,
    y,
    problem_type,
    feature_names=None,
):
    """
    Extract feature importance from tree-based models
    when available.
    """
    model = pipeline.named_steps.get("model")

    if model is None:
        return []

    if not hasattr(model, "feature_importances_"):
        return []

    try:

        if feature_names is None:
            feature_names = _get_feature_names(pipeline, X)

        importances = model.feature_importances_

        if len(importances) != len(feature_names):
            importances = importances[:len(feature_names)]

        items = []

        for name, importance in zip(feature_names, importances):

            items.append({
                "feature": name,
                "importance": round(
                    float(importance),
                    4,
                ),
            })

        items.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

        return items

    except Exception:
        return []


# ============================================================
# COEFFICIENT IMPORTANCE (linear models)
# ============================================================

def get_coefficient_importance(
    pipeline,
    feature_names,
    problem_type,
    limit=20,
):
    """
    Extract feature importance from linear model
    coefficients.

    For classification, the mean absolute coefficient
    across classes is used. For regression, the absolute
    coefficients are used directly.
    """
    model = pipeline.named_steps.get("model")

    if model is None:
        return []

    if not hasattr(model, "coef_"):
        return []

    try:

        coef = np.asarray(model.coef_, dtype=float)

        if coef.ndim == 1:
            values = np.abs(coef)
        else:
            values = np.abs(coef).mean(axis=0)

        items = []

        for name, value in zip(feature_names, values):

            items.append({
                "feature": name,
                "importance": round(
                    float(value),
                    4,
                ),
            })

        items.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

        return items[:limit]

    except Exception:
        return []


# ============================================================
# PERMUTATION IMPORTANCE (fallback)
# ============================================================

def get_permutation_importance(
    pipeline,
    X,
    y,
    problem_type,
    feature_names,
    n_repeats=5,
    limit=20,
    sample_size=500,
):
    """
    Compute permutation importance on a sample. Used when
    SHAP is not applicable and the model has neither tree
    importances nor coefficients.
    """
    try:

        n = len(X)

        sample = min(sample_size, n)

        X_sample = X.iloc[:sample]
        y_sample = y.iloc[:sample]

        scoring = (
            "f1_macro"
            if problem_type == "Classification"
            else "r2"
        )

        result = permutation_importance(
            pipeline,
            X_sample,
            y_sample,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
        )

        items = []

        for i, name in enumerate(feature_names):

            if i >= len(result.importances_mean):
                break

            items.append({
                "feature": name,
                "importance": round(
                    float(
                        np.abs(result.importances_mean[i])
                    ),
                    4,
                ),
            })

        items.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

        return items[:limit]

    except Exception:
        return []


# ============================================================
# SHAP ANALYSIS
# ============================================================

def compute_shap_values(
    pipeline,
    X,
    problem_type,
    feature_names=None,
    sample_size=150,
):
    """
    Compute SHAP values for the model.

    Uses the full pipeline so input data flows through
    preprocessing exactly as during training.
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

        if feature_names is None:
            feature_names = _get_feature_names(pipeline, X)

        # SHAP needs dense input; cap for sparse TF-IDF.
        if hasattr(X_transformed, "toarray"):

            # Sample before densifying to avoid huge arrays
            n = X_transformed.shape[0]
            sample_idx = np.random.RandomState(42).choice(
                n,
                size=min(sample_size, n),
                replace=False,
            )

            X_sample_sparse = X_transformed[sample_idx]

            X_sample_dense = X_sample_sparse.toarray()

            X_df = pd.DataFrame(
                X_sample_dense,
                columns=feature_names,
            )

        else:

            n = len(X)
            sample_idx = np.random.RandomState(42).choice(
                n,
                size=min(sample_size, n),
                replace=False,
            )

            X_df = pd.DataFrame(
                X_transformed,
                columns=feature_names,
            ).iloc[sample_idx].reset_index(drop=True)

        is_tree = isinstance(
            model,
            (
                __import__(
                    "sklearn.ensemble",
                    fromlist=["*"],
                ).RandomForestClassifier,
                __import__(
                    "sklearn.ensemble",
                    fromlist=["*"],
                ).RandomForestRegressor,
                __import__(
                    "sklearn.tree",
                    fromlist=["*"],
                ).DecisionTreeClassifier,
                __import__(
                    "sklearn.tree",
                    fromlist=["*"],
                ).DecisionTreeRegressor,
            ),
        )

        if is_tree:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X_df)

        return shap_values, X_df, explainer

    except Exception:
        return None, [], None


# ============================================================
# SHAP SUMMARIZATION
# ============================================================

def summarize_shap(
    shap_values,
    feature_names,
    problem_type,
    limit=20,
):
    """
    Convert raw SHAP values into a readable per-feature
    summary, limited to the top features.
    """
    try:

        values = np.asarray(shap_values, dtype=float)

        if values.ndim == 3:
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
                    4,
                ),
            })

        items.sort(
            key=lambda item: item["mean_abs_shap"],
            reverse=True,
        )

        return items[:limit]

    except Exception:
        return []


# ============================================================
# SHAP BAR CHART
# ============================================================

def generate_shap_chart(
    shap_values,
    X_sample,
    problem_type,
    output_path,
):
    """
    Generate a SHAP summary bar chart, limited to a small
    number of features.
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
            max_display=15,
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=100,
            bbox_inches="tight",
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
    chart_id=None,
    schema=None,
):
    """
    Explain the best model.

    Uses SHAP where compatible; otherwise falls back to
    coefficients (linear) or permutation importance.

    Returns
    -------
    dict
        success, message, method, top_features,
        feature_importance, shap_summary, chart_file,
        source_columns.
    """

    result = {
        "success": False,
        "message": "",
        "method": None,
        "shap_summary": [],
        "feature_importance": [],
        "chart_file": None,
        "top_features": [],
        "source_columns": {},
    }

    if pipeline is None:
        result["message"] = "No trained model was provided."
        return result

    if schema is None:
        schema = analyze_schema(
            df,
            target_column=target_column,
        )

    text_columns = schema["text_columns"]

    feature_names = _get_feature_names(pipeline, X)

    # ------------------------------------------------
    # SHAP
    # ------------------------------------------------

    shap_values, X_sample, explainer = compute_shap_values(
        pipeline,
        X,
        problem_type,
        feature_names=feature_names,
    )

    if shap_values is not None:

        shap_summary = summarize_shap(
            shap_values,
            feature_names,
            problem_type,
        )

        result["shap_summary"] = shap_summary

        result["top_features"] = [
            item["feature"]
            for item in shap_summary[:10]
        ]

        result["method"] = "SHAP"

        if charts_dir:

            os.makedirs(
                charts_dir,
                exist_ok=True,
            )

            base = chart_id or "shap"

            chart_path = os.path.join(
                charts_dir,
                f"{base}_shap.png",
            )

            if generate_shap_chart(
                shap_values,
                X_sample,
                problem_type,
                chart_path,
            ):

                result["chart_file"] = os.path.basename(
                    chart_path
                )

        result["success"] = True

        result["message"] = (
            "SHAP analysis completed successfully. "
            "Feature contributions represent learned "
            "associations, not proof of causation."
        )

    # ------------------------------------------------
    # COEFFICIENTS (fallback for linear models)
    # ------------------------------------------------

    if not result["feature_importance"]:

        coefficient_importance = get_coefficient_importance(
            pipeline,
            feature_names,
            problem_type,
        )

        if coefficient_importance:

            result["feature_importance"] = (
                coefficient_importance
            )

            result["top_features"] = [
                item["feature"]
                for item in coefficient_importance[:10]
            ]

            if result["method"] is None:
                result["method"] = "Model Coefficients"

    # ------------------------------------------------
    # TREE IMPORTANCE
    # ------------------------------------------------

    if not result["feature_importance"]:

        tree_importance = get_model_feature_importance(
            pipeline,
            X,
            y,
            problem_type,
            feature_names=feature_names,
        )

        if tree_importance:

            result["feature_importance"] = tree_importance

            result["top_features"] = [
                item["feature"]
                for item in tree_importance[:10]
            ]

            if result["method"] is None:
                result["method"] = "Feature Importance"

    # ------------------------------------------------
    # PERMUTATION IMPORTANCE (final fallback)
    # ------------------------------------------------

    if not result["feature_importance"]:

        permutation = get_permutation_importance(
            pipeline,
            X,
            y,
            problem_type,
            feature_names,
        )

        if permutation:

            result["feature_importance"] = permutation

            result["top_features"] = [
                item["feature"]
                for item in permutation[:10]
            ]

            result["method"] = "Permutation Importance"

    # ------------------------------------------------
    # SOURCE COLUMN MAP
    # ------------------------------------------------

    for name in result["top_features"]:
        source = _source_column(name, text_columns)
        result["source_columns"][name] = (
            source if source else "structured"
        )

    if not result["shap_summary"] and not result["feature_importance"]:

        result["message"] = (
            "Explainability could not be computed for "
            "this model."
        )

    else:

        result["success"] = True

    return result