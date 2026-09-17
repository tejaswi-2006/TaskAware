import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from modules.preprocessing import (
    build_preprocessing_pipeline,
    prepare_target
)
from modules.model_evaluator import (
    classification_metrics,
    regression_metrics,
    roc_auc_from_scores
)


# ============================================================
# MODEL REGISTRY
# ============================================================

def get_classification_models():
    """
    Return the classification models TaskAware evaluates.
    """

    return [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            "supports_probability": True
        },
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier(
                max_depth=None,
                random_state=42
            ),
            "supports_probability": True
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            "supports_probability": True
        }
    ]


def get_regression_models():
    """
    Return the regression models TaskAware evaluates.
    """

    return [
        {
            "name": "Linear Regression",
            "estimator": LinearRegression(),
            "supports_probability": False
        },
        {
            "name": "Ridge Regression",
            "estimator": Ridge(
                alpha=1.0,
                random_state=42
            ),
            "supports_probability": False
        },
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeRegressor(
                max_depth=None,
                random_state=42
            ),
            "supports_probability": False
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            "supports_probability": False
        }
    ]


# ============================================================
# CROSS VALIDATION
# ============================================================

def cross_validate_model(
    preprocessor,
    X,
    y,
    model_config,
    problem_type,
    n_splits=5
):
    """
    Run k-fold cross-validation using the full pipeline
    (preprocessing + model). The preprocessing is refit
    inside every fold, which keeps the evaluation
    leakage-free.
    """

    if problem_type == "Classification":

        fold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

    else:

        fold = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

    scoring = (
        "f1_macro"
        if problem_type == "Classification"
        else "r2"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model_config["estimator"])
        ]
    )

    try:

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=fold,
            scoring=scoring,
            n_jobs=-1
        )

        return {
            "mean": round(
                float(np.mean(scores)),
                4
            ),
            "std": round(
                float(np.std(scores)),
                4
            ),
            "all_folds": [
                round(float(s), 4)
                for s in scores
            ]
        }

    except Exception:

        return {
            "mean": None,
            "std": None,
            "all_folds": []
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
    n_classes=1
):
    """
    Build a full pipeline (preprocessing + model), fit it
    on the training fold only, and evaluate on the test
    fold.
    """

    numerical = [
        c for c in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[c])
    ]

    categorical = [
        c for c in X_train.columns
        if not pd.api.types.is_numeric_dtype(X_train[c])
    ]

    preprocessor = build_preprocessing_pipeline(
        numerical,
        categorical
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model_config["estimator"])
        ]
    )

    # Fit on training data ONLY
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # ------------------------------------------------
    # METRICS
    # ------------------------------------------------

    if problem_type == "Classification":

        metrics = classification_metrics(
            y_test,
            y_pred,
            classes=np.unique(y_train)
        )

        if (
            model_config["supports_probability"]
            and n_classes >= 2
        ):

            y_proba = pipeline.predict_proba(X_test)

            if y_proba.shape[1] == n_classes:

                metrics["roc_auc"] = (
                    roc_auc_from_scores(
                        y_test,
                        y_proba,
                        classes=np.unique(y_train)
                    )
                )

    else:

        metrics = regression_metrics(
            y_test,
            y_pred
        )

    return pipeline, y_pred, metrics


# ============================================================
# CLASSIFICATION RUN
# ============================================================

def run_classification(
    df,
    target_column,
    test_size=0.2
):
    """
    Full classification workflow:
        split -> per-model CV -> test evaluation.
    """

    roles_result = _prepare_data(
        df,
        target_column,
        "Classification"
    )

    X = roles_result["X"]
    y = roles_result["y"]
    target_info = roles_result["target_info"]
    roles = roles_result["roles"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    n_classes = len(
        np.unique(y)
    )

    models = get_classification_models()

    results = []

    preprocessor = build_preprocessing_pipeline(
        [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c])
        ],
        [
            c for c in X.columns
            if not pd.api.types.is_numeric_dtype(X[c])
        ]
    )

    for config in models:

        cv = cross_validate_model(
            preprocessor,
            X_train,
            y_train,
            config,
            "Classification"
        )

        pipeline, y_pred, metrics = train_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            config,
            "Classification",
            n_classes
        )

        results.append({
            "name": config["name"],
            "cv_mean": cv["mean"],
            "cv_std": cv["std"],
            "cv_folds": cv["all_folds"],
            **metrics
        })

    return {
        "problem_type": "Classification",
        "target_column": target_column,
        "target_info": target_info,
        "roles": roles,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "n_classes": n_classes,
        "results": results,
        "models": models,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }


# ============================================================
# REGRESSION RUN
# ============================================================

def run_regression(
    df,
    target_column,
    test_size=0.2
):
    """
    Full regression workflow.
    """

    roles_result = _prepare_data(
        df,
        target_column,
        "Regression"
    )

    X = roles_result["X"]
    y = roles_result["y"]
    target_info = roles_result["target_info"]
    roles = roles_result["roles"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    models = get_regression_models()

    results = []

    preprocessor = build_preprocessing_pipeline(
        [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c])
        ],
        [
            c for c in X.columns
            if not pd.api.types.is_numeric_dtype(X[c])
        ]
    )

    for config in models:

        cv = cross_validate_model(
            preprocessor,
            X_train,
            y_train,
            config,
            "Regression"
        )

        pipeline, y_pred, metrics = train_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            config,
            "Regression"
        )

        results.append({
            "name": config["name"],
            "cv_mean": cv["mean"],
            "cv_std": cv["std"],
            "cv_folds": cv["all_folds"],
            **metrics
        })

    return {
        "problem_type": "Regression",
        "target_column": target_column,
        "target_info": target_info,
        "roles": roles,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "results": results,
        "models": models,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def _prepare_data(
    df,
    target_column,
    problem_type
):
    """
    Split the dataset into feature matrix X and target y.
    Identifier and constant columns are removed.
    """

    from modules.preprocessing import identify_column_roles

    roles = identify_column_roles(
        df,
        target_column
    )

    X = df[roles["feature_columns"]]

    y, target_info = prepare_target(
        df,
        target_column,
        problem_type
    )

    return {
        "X": X,
        "y": y,
        "target_info": target_info,
        "roles": roles
    }


# ============================================================
# TOP LEVEL ENTRY
# ============================================================

def train_models(
    df,
    target_column,
    problem_type
):
    """
    Dispatch to the correct training workflow based on
    the problem type.
    """

    if problem_type == "Classification":

        return run_classification(
            df,
            target_column
        )

    if problem_type == "Regression":

        return run_regression(
            df,
            target_column
        )

    raise ValueError(
        f"Unsupported problem type: {problem_type}"
    )