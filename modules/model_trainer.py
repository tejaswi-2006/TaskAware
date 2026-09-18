"""
Model training for TaskAware.

Uses the central schema analyzer and the model recommender to
select compatible models. Provides zero-feature protection and
clear errors instead of raw scikit-learn tracebacks.
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    TimeSeriesSplit,
    cross_val_score,
)

from modules.schema_analyzer import analyze_schema
from modules.preprocessing import (
    build_preprocessing_pipeline,
    prepare_target,
)
from modules.model_recommender import recommend_models
from modules.model_evaluator import (
    classification_metrics,
    regression_metrics,
    roc_auc_from_scores,
)


class TaskAwareTrainingError(Exception):
    """
    Raised when training cannot proceed. The message is
    user-facing and must never leak a raw traceback.
    """


# ============================================================
# CROSS VALIDATION
# ============================================================

def _build_preprocessor(schema):
    """
    Build the type-aware preprocessor from the schema.
    """
    return build_preprocessing_pipeline(
        schema["numerical_columns"],
        schema["categorical_columns"],
        schema["text_columns"],
        boolean_columns=schema["boolean_columns"],
        date_columns=schema["date_columns"],
    )


def _make_cv(problem_type, X, y=None):
    """
    Choose the cross-validation strategy based on the
    problem type.
    """
    if problem_type == "Classification":
        return StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

    if problem_type == "Regression":
        return KFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

    return KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )


def cross_validate_model(
    preprocessor,
    X,
    y,
    model_config,
    problem_type,
    n_splits=5,
):
    """
    Run k-fold cross-validation using the full pipeline
    (preprocessing + model). Preprocessing is refit inside
    every fold, which keeps the evaluation leakage-free.
    """
    if problem_type == "Classification":
        fold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )
    else:
        fold = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )

    scoring = (
        "f1_macro"
        if problem_type == "Classification"
        else "r2"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model_config["estimator"]),
        ]
    )

    try:

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=fold,
            scoring=scoring,
            n_jobs=-1,
        )

        return {
            "mean": round(
                float(np.mean(scores)),
                4,
            ),
            "std": round(
                float(np.std(scores)),
                4,
            ),
            "all_folds": [
                round(float(s), 4)
                for s in scores
            ],
        }

    except Exception:
        return {
            "mean": None,
            "std": None,
            "all_folds": [],
        }


# ============================================================
# TRAIN AND EVALUATE ONE MODEL
# ============================================================

def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    model_config,
    problem_type,
    n_classes=1,
    preprocessor=None,
):
    """
    Build a full pipeline (preprocessing + model), fit it
    on the training fold only, and evaluate on the test
    fold.
    """
    if preprocessor is None:
        schema = analyze_schema(X_train)
        preprocessor = _build_preprocessor(schema)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model_config["estimator"]),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if problem_type == "Classification":

        metrics = classification_metrics(
            y_test,
            y_pred,
            classes=np.unique(y_train),
        )

        if (
            model_config["supports_probability"]
            and n_classes >= 2
        ):

            y_proba = pipeline.predict_proba(X_test)

            if y_proba.shape[1] == n_classes:

                metrics["roc_auc"] = roc_auc_from_scores(
                    y_test,
                    y_proba,
                    classes=np.unique(y_train),
                )

    else:

        metrics = regression_metrics(
            y_test,
            y_pred,
        )

    return pipeline, y_pred, metrics


# ============================================================
# VALIDATION GUARDS
# ============================================================

def _validate_features(schema, df):
    """
    Raise a clear TaskAware error when no usable features
    remain.
    """
    usable_roles = {
        "numeric",
        "categorical",
        "boolean",
        "text",
        "datetime",
    }

    feature_records = [
        record for record in schema["columns"]
        if record["column"] in schema["feature_columns"]
    ]

    if not schema["feature_columns"]:

        detail = _describe_removed(schema, df)

        raise TaskAwareTrainingError(
            "TaskAware could not find any usable feature "
            "columns. " + detail
        )

    usable = any(
        record["role"] in usable_roles
        for record in feature_records
    )

    if not usable:
        raise TaskAwareTrainingError(
            "No usable feature columns were found."
        )


def _describe_removed(schema, df):
    """
    Explain which columns were removed and why.
    """
    parts = []

    if schema["id_columns"]:
        parts.append(
            f"Identifier columns removed: "
            f"{', '.join(schema['id_columns'])}"
        )

    if schema["constant_columns"]:
        parts.append(
            f"Constant columns removed: "
            f"{', '.join(schema['constant_columns'])}"
        )

    if schema["unsupported_columns"]:
        parts.append(
            f"Unsupported columns removed: "
            f"{', '.join(schema['unsupported_columns'])}"
        )

    if not parts:
        parts.append(
            "All non-target columns were detected as "
            "non-predictive."
        )

    return (
        "Detected columns: "
        f"{', '.join(str(c) for c in df.columns)}. "
        + " ".join(parts)
        + " Upload a dataset with at least one usable "
        "feature column."
    )


def _validate_target(df, target_column):
    """
    Validate that the target is usable for supervised
    learning.
    """
    if not target_column:
        raise TaskAwareTrainingError(
            "No target variable was provided."
        )

    if target_column not in df.columns:
        raise TaskAwareTrainingError(
            f"Target '{target_column}' does not exist in "
            "the dataset."
        )

    target = df[target_column].dropna()

    if len(target) == 0:
        raise TaskAwareTrainingError(
            "The target column contains no valid values."
        )

    if target.nunique() <= 1:
        raise TaskAwareTrainingError(
            "The target column has no meaningful "
            "variation."
        )


# ============================================================
# SPLIT STRATEGY
# ============================================================

def _split_data(X, y, problem_type, schema, test_size=0.2):
    """
    Choose the train/test split strategy.

    For time-dependent data, a chronological split is used
    so that the model never sees future data during training.
    """
    if schema["is_time_series"] and schema["time_column"]:

        time_col = schema["time_column"]

        if time_col in X.columns:

            parsed = pd.to_datetime(
                X[time_col],
                errors="coerce",
                format="mixed",
            )

            if parsed.notna().sum() >= 2:

                order = np.argsort(
                    np.argsort(
                        parsed.values.astype("datetime64[ns]")
                    )
                )

                n = len(X)
                split = int(n * (1 - test_size))

                idx = np.argsort(order)

                train_idx = idx[:split]
                test_idx = idx[split:]

                return (
                    X.iloc[train_idx],
                    X.iloc[test_idx],
                    y.iloc[train_idx],
                    y.iloc[test_idx],
                ), True

    if problem_type == "Classification":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
        )

    return (X_tr, X_te, y_tr, y_te), False


# ============================================================
# DATA PREPARATION
# ============================================================

def _prepare_data(df, target_column, problem_type):
    """
    Split the dataset into feature matrix X and target y
    using the central schema.
    """
    schema = analyze_schema(
        df,
        target_column=target_column,
    )

    _validate_features(schema, df)
    _validate_target(df, target_column)

    X = df[schema["feature_columns"]]

    y, target_info = prepare_target(
        df,
        target_column,
        problem_type,
    )

    return {
        "X": X,
        "y": y,
        "target_info": target_info,
        "schema": schema,
    }


# ============================================================
# CLASSIFICATION RUN
# ============================================================

def run_classification(
    df,
    target_column,
    test_size=0.2,
):
    """
    Full classification workflow:
        split -> per-model CV -> test evaluation.
    """
    roles_result = _prepare_data(
        df,
        target_column,
        "Classification",
    )

    X = roles_result["X"]
    y = roles_result["y"]
    target_info = roles_result["target_info"]
    schema = roles_result["schema"]

    (X_train, X_test, y_train, y_test), used_chrono = _split_data(
        X, y, "Classification", schema, test_size,
    )

    n_classes = len(np.unique(y))

    preprocessor = _build_preprocessor(schema)

    # Validate the preprocessor actually produces features
    _validate_transformed_features(
        preprocessor, X_train
    )

    models = recommend_models(
        schema,
        "Classification",
    )

    results = []

    for config in models:

        cv = cross_validate_model(
            preprocessor,
            X_train,
            y_train,
            config,
            "Classification",
        )

        pipeline, y_pred, metrics = train_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            config,
            "Classification",
            n_classes,
            preprocessor=preprocessor,
        )

        results.append({
            "name": config["name"],
            "cv_mean": cv["mean"],
            "cv_std": cv["std"],
            "cv_folds": cv["all_folds"],
            "compatible_reason": config["reason"],
            **metrics,
        })

    return {
        "problem_type": "Classification",
        "target_column": target_column,
        "target_info": target_info,
        "schema": schema,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "n_classes": n_classes,
        "results": results,
        "models": models,
        "used_chronological_split": used_chrono,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ============================================================
# REGRESSION RUN
# ============================================================

def run_regression(
    df,
    target_column,
    test_size=0.2,
):
    """
    Full regression workflow.
    """
    roles_result = _prepare_data(
        df,
        target_column,
        "Regression",
    )

    X = roles_result["X"]
    y = roles_result["y"]
    target_info = roles_result["target_info"]
    schema = roles_result["schema"]

    (X_train, X_test, y_train, y_test), used_chrono = _split_data(
        X, y, "Regression", schema, test_size,
    )

    preprocessor = _build_preprocessor(schema)

    _validate_transformed_features(
        preprocessor, X_train
    )

    models = recommend_models(
        schema,
        "Regression",
    )

    results = []

    for config in models:

        cv = cross_validate_model(
            preprocessor,
            X_train,
            y_train,
            config,
            "Regression",
        )

        pipeline, y_pred, metrics = train_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            config,
            "Regression",
            preprocessor=preprocessor,
        )

        results.append({
            "name": config["name"],
            "cv_mean": cv["mean"],
            "cv_std": cv["std"],
            "cv_folds": cv["all_folds"],
            "compatible_reason": config["reason"],
            **metrics,
        })

    return {
        "problem_type": "Regression",
        "target_column": target_column,
        "target_info": target_info,
        "schema": schema,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "results": results,
        "models": models,
        "used_chronological_split": used_chrono,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def _validate_transformed_features(preprocessor, X_train):
    """
    Fit the preprocessor on training data and verify that
    at least one feature is produced. Never pass a (rows, 0)
    matrix to a model.
    """
    try:

        transformed = preprocessor.fit_transform(X_train)

        n_features = transformed.shape[1]

        if n_features < 1:
            raise TaskAwareTrainingError(
                "The preprocessing pipeline produced zero "
                "features. No usable feature columns were "
                "found in the dataset."
            )

        return transformed

    except TaskAwareTrainingError:
        raise

    except Exception:
        raise TaskAwareTrainingError(
            "The preprocessing pipeline could not be "
            "fitted. Check that the dataset contains at "
            "least one usable feature column."
        )


# ============================================================
# TOP LEVEL ENTRY
# ============================================================

def train_models(df, target_column, problem_type):
    """
    Dispatch to the correct training workflow based on the
    problem type.
    """
    if problem_type == "Classification":
        return run_classification(df, target_column)

    if problem_type == "Regression":
        return run_regression(df, target_column)

    raise ValueError(
        f"Unsupported problem type: {problem_type}"
    )


# ============================================================
# TRAIN A SINGLE NAMED MODEL
# ============================================================

def train_named_model(
    df,
    target_column,
    problem_type,
    model_name,
    test_size=0.2,
):
    """
    Retrain a single named model so that explainability
    can use the fitted pipeline.
    """
    roles_result = _prepare_data(
        df,
        target_column,
        problem_type,
    )

    X = roles_result["X"]
    y = roles_result["y"]
    target_info = roles_result["target_info"]
    schema = roles_result["schema"]

    (X_train, X_test, y_train, y_test), _ = _split_data(
        X, y, problem_type, schema, test_size,
    )

    models = recommend_models(
        schema,
        problem_type,
    )

    config = next(
        (
            m for m in models
            if m["name"] == model_name
        ),
        models[0],
    )

    preprocessor = _build_preprocessor(schema)

    pipeline, _, _ = train_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        config,
        problem_type,
        n_classes=(
            len(np.unique(y))
            if problem_type == "Classification"
            else 1
        ),
        preprocessor=preprocessor,
    )

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "target_info": target_info,
        "schema": schema,
        "model_name": config["name"],
        "preprocessor": preprocessor,
    }