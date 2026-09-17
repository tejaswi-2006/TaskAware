import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# COLUMN CLASSIFICATION
# ============================================================

def identify_column_roles(
    df,
    target_column=None
):
    """
    Classify columns into numerical, categorical, id and
    constant groups.

    Returns
    -------
    dict
        {
            "id_columns": [...],
            "constant_columns": [...],
            "numerical_columns": [...],
            "categorical_columns": [...],
            "feature_columns": [...]
        }
    """

    id_columns = []

    constant_columns = []

    numerical_columns = []

    categorical_columns = []

    for column in df.columns:

        if column == target_column:
            continue

        unique_count = df[column].nunique(
            dropna=True
        )

        # ------------------------------------------------
        # IDENTIFIER DETECTION
        # ------------------------------------------------

        unique_ratio = (
            unique_count / len(df)
        )

        name = str(column).lower().replace("_", " ")

        id_words = [
            "id",
            "identifier",
            "customer id",
            "student id",
            "user id",
            "record id",
            "roll number",
            "serial number"
        ]

        name_suggests_id = any(
            word in name
            for word in id_words
        )

        is_numeric = pd.api.types.is_numeric_dtype(
            df[column]
        )

        if name_suggests_id:
            id_columns.append(column)
            continue

        # Near-unique non-numeric key
        if (
            unique_ratio >= 0.95
            and not is_numeric
        ):
            id_columns.append(column)
            continue

        # ------------------------------------------------
        # CONSTANT
        # ------------------------------------------------

        if unique_count <= 1:
            constant_columns.append(column)
            continue

        # ------------------------------------------------
        # NUMERICAL / CATEGORICAL
        # ------------------------------------------------

        if pd.api.types.is_numeric_dtype(df[column]):
            numerical_columns.append(column)
        else:
            categorical_columns.append(column)

    feature_columns = (
        numerical_columns + categorical_columns
    )

    return {
        "id_columns": id_columns,
        "constant_columns": constant_columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "feature_columns": feature_columns
    }


# ============================================================
# MISSING VALUE SUMMARY
# ============================================================

def missing_value_summary(df, columns):
    """
    Summarize missing values for the given columns.
    """

    rows = []

    total = len(df)

    for column in columns:

        missing_count = int(
            df[column].isnull().sum()
        )

        missing_ratio = (
            missing_count / total
            if total > 0
            else 0
        )

        rows.append({
            "column": column,
            "missing_count": missing_count,
            "missing_ratio": round(
                missing_ratio,
                3
            ),
            "strategy": (
                "Median imputation"
                if pd.api.types.is_numeric_dtype(
                    df[column]
                )
                else "Most frequent imputation"
            )
        })

    rows.sort(
        key=lambda item: item["missing_ratio"],
        reverse=True
    )

    return rows


# ============================================================
# PREPROCESSING PIPELINE (LEAKAGE-SAFE)
# ============================================================

def build_preprocessing_pipeline(
    numerical_columns,
    categorical_columns,
    one_hot_drop=None
):
    """
    Build a scikit-learn ColumnTransformer.

    The transformer must only be FITTED on training data.
    The test fold is transformed using the already-fitted
    transformer. This prevents data leakage.

    Numerical strategy:
        median imputation -> standard scaling

    Categorical strategy:
        most-frequent imputation -> one-hot encoding
    """

    transformers = []

    if numerical_columns:

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        )

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop=one_hot_drop
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )


# ============================================================
# TARGET ENCODING
# ============================================================

def prepare_target(
    df,
    target_column,
    problem_type
):
    """
    Extract the target variable from the dataset.

    For classification, string labels are encoded to
    integers so scikit-learn models can use them.

    For clustering, there is no target.
    """

    if not target_column:
        return None, None

    target = df[target_column]

    mapping = None

    if problem_type == "Classification":

        unique_labels = target.dropna().unique()

        sorted_labels = sorted(
            unique_labels,
            key=str
        )

        mapping = {
            label: index
            for index, label
            in enumerate(sorted_labels)
        }

        encoded = target.map(
            mapping
        )

        return encoded, {
            "mapping": mapping,
            "classes": [
                str(label)
                for label in sorted_labels
            ],
            "inverse_mapping": {
                index: str(label)
                for label, index
                in mapping.items()
            }
        }

    return target, {
        "mapping": None,
        "classes": None,
        "inverse_mapping": None
    }


# ============================================================
# FEATURE TRANSFORMATION PREVIEW
# ============================================================

def preview_transformation(
    df,
    pipeline,
    numerical_columns,
    categorical_columns
):
    """
    Fit the pipeline on the full feature matrix to show
    what the transformed data looks like.

    NOTE: This is only a preview for display purposes.
    During real training the pipeline is fitted on the
    training fold only, never on the full dataset.
    """

    feature_columns = (
        numerical_columns + categorical_columns
    )

    X = df[feature_columns]

    fitted = pipeline.fit(X)

    X_transformed = fitted.transform(X)

    feature_names = fitted.get_feature_names_out().tolist()

    return {
        "rows": int(X_transformed.shape[0]),
        "features_before": len(feature_columns),
        "features_after": int(X_transformed.shape[1]),
        "feature_names": feature_names,
        "is_sparse": hasattr(
            X_transformed,
            "toarray"
        )
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def prepare_preprocessing(
    df,
    target_column,
    problem_type="Classification"
):
    """
    Build the complete preprocessing plan for the dataset.

    This does NOT fit anything on the full dataset for
    real training. It returns the pipeline and a report
    that the UI can display.
    """

    if problem_type == "Clustering":
        target_column = None

    roles = identify_column_roles(
        df,
        target_column
    )

    numerical_columns = roles["numerical_columns"]
    categorical_columns = roles["categorical_columns"]

    pipeline = build_preprocessing_pipeline(
        numerical_columns,
        categorical_columns
    )

    missing_summary = missing_value_summary(
        df,
        roles["feature_columns"]
    )

    target, target_info = prepare_target(
        df,
        target_column,
        problem_type
    )

    preview = preview_transformation(
        df,
        pipeline,
        numerical_columns,
        categorical_columns
    )

    # Describe the actual steps used
    steps = []

    if numerical_columns:
        steps.append(
            f"Numerical features ({len(numerical_columns)}): "
            "median imputation + standard scaling"
        )

    if categorical_columns:
        steps.append(
            f"Categorical features ({len(categorical_columns)}): "
            "most-frequent imputation + one-hot encoding"
        )

    if roles["id_columns"]:
        steps.append(
            f"Identifier columns removed "
            f"({len(roles['id_columns'])}): "
            f"{', '.join(roles['id_columns'])}"
        )

    if roles["constant_columns"]:
        steps.append(
            f"Constant columns removed "
            f"({len(roles['constant_columns'])}): "
            f"{', '.join(roles['constant_columns'])}"
        )

    if target_column:
        steps.append(
            "Target handled separately from feature "
            "preprocessing"
        )

    else:
        steps.append(
            "Clustering mode: all prepared columns are "
            "used as features"
        )

    return {
        "target_column": target_column,
        "roles": roles,
        "pipeline": pipeline,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "missing_summary": missing_summary,
        "target_info": target_info,
        "preview": preview,
        "steps": steps,
        "n_rows": len(df),
        "has_missing": any(
            item["missing_count"] > 0
            for item in missing_summary
        )
    }