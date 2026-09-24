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
    GroupKFold,
    GroupShuffleSplit,
    TimeSeriesSplit,
    cross_val_score,
)

from modules.schema_analyzer import analyze_schema
from modules.preprocessing import (
    build_preprocessing_pipeline,
    feature_accounting,
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


def _group_values(groups):
    """
    Return group labels as an array, or None when groups
    are unusable for splitting.
    """
    if groups is None:
        return None

    try:
        values = np.asarray(groups)
    except Exception:
        return None

    if len(values) == 0:
        return None

    try:
        n_groups = len(pd.unique(values))
    except Exception:
        return None

    if n_groups < 2:
        return None

    return values


def cross_validate_model(
    preprocessor,
    X,
    y,
    model_config,
    problem_type,
    n_splits=5,
    groups=None,
):
    """
    Run k-fold cross-validation using the full pipeline
    (preprocessing + model). Preprocessing is refit inside
    every fold, which keeps the evaluation leakage-free.

    When repeated-entity groups are provided, GroupKFold
    keeps each entity within a single fold.
    """
    group_values = _group_values(groups)

    if group_values is not None and len(
        pd.unique(group_values)
    ) >= n_splits:
        fold = GroupKFold(n_splits=n_splits)
    elif problem_type == "Classification":
        fold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )
        group_values = None
    else:
        fold = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )
        group_values = None

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

        import warnings as _warnings

        with _warnings.catch_warnings():

            # sklearn emits benign "Pipeline not fitted yet"
            # FutureWarnings when validating transform
            # columns; suppress to keep output clean.
            _warnings.simplefilter("ignore", FutureWarning)
            _warnings.simplefilter("ignore", UserWarning)

            if group_values is not None:
                scores = cross_val_score(
                    pipeline,
                    X,
                    y,
                    groups=group_values,
                    cv=fold,
                    scoring=scoring,
                    n_jobs=-1,
                )
            else:
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

def _split_groups(df, X, schema):
    """
    Return the group labels aligned with X, or None when no
    usable group column exists. The group column itself is
    usually excluded from features, so labels come from df.
    """
    group_columns = schema.get("group_columns") or []

    if not group_columns:
        return None, None

    column = group_columns[0]

    if column not in df.columns:
        return None, None

    try:
        groups = df.loc[X.index, column]
    except Exception:
        return None, None

    if _group_values(groups) is None:
        return None, None

    return groups, column


def _split_data(
    X,
    y,
    problem_type,
    schema,
    test_size=0.2,
    groups=None,
    group_column=None,
):
    """
    Choose the train/test split strategy.

    For time-dependent data, a chronological split is used
    so that the model never sees future data during training.

    When repeated-entity groups are provided, whole entities
    are kept within train or test (GroupShuffleSplit) so
    that related records never leak across the split.

    Returns (splits, info) where info carries
    {"chronological": bool, "group": column or None}.
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
                    y.iloc[train_idx],
                ), {
                    "chronological": True,
                    "group": None,
                }

    group_values = _group_values(groups)

    if group_values is not None:
        try:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=42,
            )

            train_idx, test_idx = next(
                splitter.split(X, y, groups=group_values)
            )

            return (
                X.iloc[train_idx],
                X.iloc[test_idx],
                y.iloc[train_idx],
                y.iloc[test_idx],
            ), {
                "chronological": False,
                "group": group_column,
            }
        except Exception:
            pass

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

    return (X_tr, X_te, y_tr, y_te), {
        "chronological": False,
        "group": None,
    }


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

    split_groups, split_group_column = _split_groups(
        df, X, schema
    )

    (X_train, X_test, y_train, y_test), split_info = _split_data(
        X, y, "Classification", schema, test_size,
        groups=split_groups,
        group_column=split_group_column,
    )

    if split_groups is not None:
        cv_groups = split_groups.loc[X_train.index]
    else:
        cv_groups = None

    n_classes = len(np.unique(y))

    preprocessor = _build_preprocessor(schema)

    # Validate the preprocessor actually produces features
    transformed_preview = _validate_transformed_features(
        preprocessor, X_train
    )

    n_transformed = int(transformed_preview.shape[1])

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
            groups=cv_groups,
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
        "used_chronological_split": split_info["chronological"],
        "used_group_split": split_info["group"],
        "n_transformed_features": n_transformed,
        "feature_accounting": feature_accounting(
            df,
            target_column,
            schema,
            {"features_after": n_transformed},
        ),
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

    split_groups, split_group_column = _split_groups(
        df, X, schema
    )

    (X_train, X_test, y_train, y_test), split_info = _split_data(
        X, y, "Regression", schema, test_size,
        groups=split_groups,
        group_column=split_group_column,
    )

    if split_groups is not None:
        cv_groups = split_groups.loc[X_train.index]
    else:
        cv_groups = None

    preprocessor = _build_preprocessor(schema)

    transformed_preview = _validate_transformed_features(
        preprocessor, X_train
    )

    n_transformed = int(transformed_preview.shape[1])

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
            groups=cv_groups,
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
        "used_chronological_split": split_info["chronological"],
        "used_group_split": split_info["group"],
        "n_transformed_features": n_transformed,
        "feature_accounting": feature_accounting(
            df,
            target_column,
            schema,
            {"features_after": n_transformed},
        ),
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

    split_groups, split_group_column = _split_groups(
        df, X, schema
    )

    (X_train, X_test, y_train, y_test), split_info = _split_data(
        X, y, problem_type, schema, test_size,
        groups=split_groups,
        group_column=split_group_column,
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
        "used_chronological_split": split_info["chronological"],
        "used_group_split": split_info["group"],
    }