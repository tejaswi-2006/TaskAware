"""
Model recommender for TaskAware.

Chooses which models are compatible with the dataset schema
and the problem type, and explains why each model is (or is
not) included.

For text datasets, only sparse-compatible linear models are
used by default, because tree/ensemble models are slow and
memory-hungry on large TF-IDF matrices.

Environment configuration (all optional; defaults preserve
existing behavior exactly):

    TASKAWARE_RF_TREES
        Number of trees for Random Forest estimators.
        Must be an integer >= 1. Default 100. Invalid
        values fall back to 100 with a visible note.

    TASKAWARE_RF_JOBS
        Parallel workers for Random Forest estimators.
        Must be -1 (all CPUs) or an integer >= 1.
        Default -1. Invalid values fall back to -1 with
        a visible note.
"""

import os


DEFAULT_RF_TREES = 100
DEFAULT_RF_JOBS = -1


def _parse_positive_int(raw, minimum=1):
    """
    Parse a strictly positive integer from an environment
    string. Returns None for missing, non-integer, boolean-
    style or out-of-range values (including 0, negatives
    and floats such as "10.5").
    """
    if raw is None:
        return None

    text = str(raw).strip()

    if not text:
        return None

    if text[0] in ("+", "-"):
        if text[0] == "-":
            return None
        text = text[1:]

    if not text.isdigit():
        return None

    try:
        value = int(text)
    except (TypeError, ValueError):
        return None

    if value < minimum:
        return None

    return value


def resolve_rf_config():
    """
    Resolve the Random Forest configuration from the
    environment without changing defaults.

    Returns
    -------
    dict
        n_estimators, n_jobs, warnings (possibly empty),
        overridden (True when an env var was applied).
    """
    warnings = []
    overridden = False

    trees_raw = os.environ.get("TASKAWARE_RF_TREES")
    jobs_raw = os.environ.get("TASKAWARE_RF_JOBS")

    n_estimators = DEFAULT_RF_TREES
    n_jobs = DEFAULT_RF_JOBS

    if trees_raw is not None and str(trees_raw).strip() != "":
        parsed = _parse_positive_int(trees_raw, minimum=1)

        if parsed is None:
            warnings.append(
                "TASKAWARE_RF_TREES="
                f"{trees_raw!r} is invalid; using default "
                f"{DEFAULT_RF_TREES}."
            )
        else:
            n_estimators = parsed
            overridden = True

    if jobs_raw is not None and str(jobs_raw).strip() != "":
        text = str(jobs_raw).strip()

        if text == "-1":
            n_jobs = -1
            overridden = True
        else:
            parsed = _parse_positive_int(text, minimum=1)

            if parsed is None:
                warnings.append(
                    "TASKAWARE_RF_JOBS="
                    f"{jobs_raw!r} is invalid; using default "
                    f"{DEFAULT_RF_JOBS}."
                )
            else:
                n_jobs = parsed
                overridden = True

    return {
        "n_estimators": n_estimators,
        "n_jobs": n_jobs,
        "warnings": warnings,
        "overridden": overridden,
    }


def _rf_reason(base_reason):
    """
    Append any RF configuration fallback warnings to the
    model reason so misconfiguration stays visible in the
    existing UI without template changes.
    """
    notes = resolve_rf_config()["warnings"]

    if not notes:
        return base_reason

    return base_reason + " " + " ".join(notes)


def _rf_classifier():
    """
    Random Forest classifier honoring TASKAWARE_RF_TREES
    and TASKAWARE_RF_JOBS (defaults 100 / -1).
    """
    from sklearn.ensemble import RandomForestClassifier

    config = resolve_rf_config()

    return RandomForestClassifier(
        n_estimators=config["n_estimators"],
        random_state=42,
        n_jobs=config["n_jobs"],
    )


def _rf_regressor():
    """
    Random Forest regressor honoring TASKAWARE_RF_TREES
    and TASKAWARE_RF_JOBS (defaults 100 / -1).
    """
    from sklearn.ensemble import RandomForestRegressor

    config = resolve_rf_config()

    return RandomForestRegressor(
        n_estimators=config["n_estimators"],
        random_state=42,
        n_jobs=config["n_jobs"],
    )


def recommend_models(
    schema,
    problem_type,
    include_text_models=True,
):
    """
    Return the list of models to evaluate for the given
    schema and problem type.

    Each entry contains:
        name, estimator factory, supports_probability,
        compatible, reason, priority
    """

    has_text = bool(schema["text_columns"])
    has_numeric = bool(schema["numerical_columns"])

    models = []

    if problem_type == "Classification":

        models = _classification_models(
            has_text,
            has_numeric,
        )

    elif problem_type == "Regression":

        models = _regression_models(
            has_text,
            has_numeric,
        )

    return [
        m for m in models
        if m["compatible"]
    ]


def explain_exclusions(
    schema,
    problem_type,
):
    """
    Explain why certain models are excluded, for the
    report / UI.
    """

    has_text = bool(schema["text_columns"])

    notes = []

    if problem_type == "Classification" and has_text:

        notes.append(
            "Decision Tree and Random Forest are excluded "
            "for text data because tree ensembles are slow "
            "and memory-intensive on large sparse TF-IDF "
            "matrices. Logistic Regression and Linear SVM "
            "scale efficiently to sparse features."
        )

    return notes


def _classification_models(has_text, has_numeric):

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    models = [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
            "supports_probability": True,
            "compatible": True,
            "reason": (
                "Works with dense and sparse features."
                if has_text
                else "Strong baseline for classification."
            ),
            "priority": 1,
        },
        {
            "name": "Linear SVM",
            "estimator": LinearSVC(
                max_iter=2000,
                random_state=42,
            ),
            "supports_probability": False,
            "compatible": has_text,
            "reason": (
                "Efficient on sparse text features."
                if has_text
                else "Not used for non-text data."
            ),
            "priority": 2,
        },
        {
            "name": "Naive Bayes",
            "estimator": MultinomialNB(),
            "supports_probability": True,
            "compatible": has_text and not has_numeric,
            "reason": (
                "Works well on non-negative TF-IDF counts."
                if has_text and not has_numeric
                else (
                    "Not used because the dataset mixes "
                    "numerical (scaled, possibly negative) "
                    "features with text."
                )
            ),
            "priority": 3,
        },
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier(
                max_depth=None,
                random_state=42,
            ),
            "supports_probability": True,
            "compatible": not has_text,
            "reason": (
                "Not used for text data (slow on sparse "
                "TF-IDF matrices)."
                if has_text
                else "Interpretable tree classifier."
            ),
            "priority": 4,
        },
        {
            "name": "Random Forest",
            "estimator": _rf_classifier(),
            "supports_probability": True,
            "compatible": not has_text,
            "reason": (
                "Not used for text data (slow on sparse "
                "TF-IDF matrices)."
                if has_text
                else _rf_reason("Strong ensemble classifier.")
            ),
            "priority": 5,
        },
    ]

    return models


def _regression_models(has_text, has_numeric):

    from sklearn.linear_model import (
        LinearRegression,
        Ridge,
    )
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    models = [
        {
            "name": "Linear Regression",
            "estimator": LinearRegression(),
            "supports_probability": False,
            "compatible": True,
            "reason": (
                "Handles sparse text features."
                if has_text
                else "Standard linear baseline."
            ),
            "priority": 1,
        },
        {
            "name": "Ridge Regression",
            "estimator": Ridge(
                alpha=1.0,
                random_state=42,
            ),
            "supports_probability": False,
            "compatible": True,
            "reason": (
                "Regularized linear model, efficient on "
                "sparse text features."
                if has_text
                else "Regularized linear baseline."
            ),
            "priority": 2,
        },
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeRegressor(
                max_depth=None,
                random_state=42,
            ),
            "supports_probability": False,
            "compatible": not has_text,
            "reason": (
                "Not used for text data."
                if has_text
                else "Interpretable tree regressor."
            ),
            "priority": 3,
        },
        {
            "name": "Random Forest",
            "estimator": _rf_regressor(),
            "supports_probability": False,
            "compatible": not has_text,
            "reason": (
                "Not used for text data."
                if has_text
                else _rf_reason("Strong ensemble regressor.")
            ),
            "priority": 4,
        },
    ]

    return models