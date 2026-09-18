"""
Model recommender for TaskAware.

Chooses which models are compatible with the dataset schema
and the problem type, and explains why each model is (or is
not) included.

For text datasets, only sparse-compatible linear models are
used by default, because tree/ensemble models are slow and
memory-hungry on large TF-IDF matrices.
"""


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
            "estimator": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            "supports_probability": True,
            "compatible": not has_text,
            "reason": (
                "Not used for text data (slow on sparse "
                "TF-IDF matrices)."
                if has_text
                else "Strong ensemble classifier."
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
            "estimator": RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            "supports_probability": False,
            "compatible": not has_text,
            "reason": (
                "Not used for text data."
                if has_text
                else "Strong ensemble regressor."
            ),
            "priority": 4,
        },
    ]

    return models