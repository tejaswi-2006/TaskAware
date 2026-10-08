"""
Preprocessing pipeline for TaskAware.

Builds a leakage-safe scikit-learn ColumnTransformer using the
central schema analyzer. Supports:

    - numeric      : median imputation + standard scaling
    - categorical  : most-frequent imputation + one-hot encoding
    - boolean      : imputation + binary conversion
    - text         : TF-IDF (one vectorizer per text column so
                     term names keep their source column)
    - datetime     : date-component extraction (year, month, day,
                     day-of-week, hour)

The transformer is only ever FITTED on training data. The test
fold is transformed using the already-fitted transformer, which
prevents data leakage.
"""

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.schema_analyzer import analyze_schema


# ============================================================
# MISSING-TARGET ROW HANDLING
# ============================================================

def drop_missing_target_rows(df, target_column):
    """
    Create the modelling dataset by removing rows whose
    selected target is missing.

    Missing target values are NEVER imputed: a row without
    a label cannot teach a supervised model anything.

    The original dataset is left unchanged; callers keep it
    for reporting and EDA and use the cleaned frame for
    preprocessing, splitting, training and evaluation.

    Returns (df_clean, row_info) where row_info carries
    rows_uploaded, rows_available_for_modeling,
    rows_excluded_missing_target, target_missing_count and
    target_warning (None when nothing was excluded).
    """
    rows_uploaded = int(len(df))

    if not target_column or target_column not in df.columns:
        return df, {
            "rows_uploaded": rows_uploaded,
            "rows_available_for_modeling": rows_uploaded,
            "rows_excluded_missing_target": 0,
            "target_missing_count": 0,
            "target_warning": None,
        }

    mask = df[target_column].isnull()
    excluded = int(mask.sum())

    if excluded == 0:
        return df, {
            "rows_uploaded": rows_uploaded,
            "rows_available_for_modeling": rows_uploaded,
            "rows_excluded_missing_target": 0,
            "target_missing_count": 0,
            "target_warning": None,
        }

    cleaned = df.loc[~mask].reset_index(drop=True)

    return cleaned, {
        "rows_uploaded": rows_uploaded,
        "rows_available_for_modeling": int(len(cleaned)),
        "rows_excluded_missing_target": excluded,
        "target_missing_count": excluded,
        "target_warning": (
            f"{excluded} rows were excluded because the "
            f"selected target contains missing values. "
            f"Target values were not imputed."
        ),
    }


# ============================================================
# TEXT COLUMN DETECTION (kept for backward compatibility)
# ============================================================

def is_text_column(series, sample_size=200):
    """
    Backward-compatible wrapper. Prefer analyze_schema().
    """
    from modules.schema_analyzer import _has_natural_language

    return _has_natural_language(series)


def _flatten_text(X):
    """
    Convert a (n, 1) text column slice into a flat list of
    strings so TfidfVectorizer can consume it.
    """
    arr = np.asarray(X).ravel()

    return [
        str(v) if v is not None and not pd.isna(v)
        else ""
        for v in arr
    ]


class _PrefixedTfidf(BaseEstimator, TransformerMixin):
    """
    Apply TF-IDF to a single text column and prefix the
    generated term names with the source column name, e.g.
        review -> review:excellent

    Keeping one transformer per text column preserves the
    connection between generated terms and their source.
    """

    def __init__(self, column, max_features=5000):
        self.column = column
        self.max_features = max_features
        self._tfidf = TfidfVectorizer(
            max_features=self.max_features,
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

    def fit(self, X, y=None):
        flat = _flatten_text(X)
        self._tfidf.fit(flat)
        return self

    def transform(self, X):
        flat = _flatten_text(X)
        return self._tfidf.transform(flat)

    def get_feature_names_out(self, input_features=None):
        names = self._tfidf.get_feature_names_out()
        return np.array(
            [f"{self.column}:{name}" for name in names]
        )


class _DateComponents(BaseEstimator, TransformerMixin):
    """
    Extract useful components from date/datetime columns.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        records = []

        for i in range(arr.shape[1]):
            col = arr[:, i]

            parsed = pd.to_datetime(
                pd.Series(col),
                errors="coerce",
                format="mixed",
            )

            out = np.zeros((len(parsed), 5))

            valid = parsed.notna()

            out[valid, 0] = parsed[valid].dt.year
            out[valid, 1] = parsed[valid].dt.month
            out[valid, 2] = parsed[valid].dt.day
            out[valid, 3] = parsed[valid].dt.dayofweek
            out[valid, 4] = parsed[valid].dt.hour

            records.append(out)

        if len(records) == 1:
            return records[0]

        return np.hstack(records)

    def get_feature_names_out(self, input_features=None):
        n = (
            len(input_features)
            if input_features is not None
            else 1
        )

        names = []

        for i in range(n):
            name = (
                str(input_features[i])
                if input_features is not None
                else f"col{i}"
            )
            names.extend(
                [
                    f"date_{name}_year",
                    f"date_{name}_month",
                    f"date_{name}_day",
                    f"date_{name}_dayofweek",
                    f"date_{name}_hour",
                ]
            )

        return np.array(names)


def _date_feature_names(columns):
    names = []
    for column in columns:
        for suffix in ("year", "month", "day", "dayofweek", "hour"):
            names.append(f"date_{column}_{suffix}")
    return names


# ============================================================
# COLUMN CLASSIFICATION (backward-compatible wrapper)
# ============================================================

def identify_column_roles(
    df,
    target_column=None
):
    """
    Backward-compatible wrapper around the central schema
    analyzer.
    """
    schema = analyze_schema(
        df,
        target_column=target_column,
    )

    return {
        "id_columns": schema["id_columns"],
        "constant_columns": schema["constant_columns"],
        "numerical_columns": schema["numerical_columns"],
        "categorical_columns": (
            schema["categorical_columns"]
            + schema["boolean_columns"]
        ),
        "text_columns": schema["text_columns"],
        "date_columns": schema["date_columns"],
        "unsupported_columns": schema["unsupported_columns"],
        "feature_columns": schema["feature_columns"],
    }


# ============================================================
# MISSING VALUE SUMMARY
# ============================================================

def missing_value_summary(df, columns, schema=None):
    """
    Summarize missing values for the given columns.
    """
    if schema is None:
        schema = analyze_schema(df)

    roles = {
        record["column"]: record["role"]
        for record in schema["columns"]
    }

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

        role = roles.get(str(column), "categorical")

        strategy = {
            "numeric": "Median imputation",
            "boolean": "Most frequent imputation",
            "categorical": "Most frequent imputation",
            "text": "Empty string + TF-IDF",
            "datetime": "Dropped rows / median reference",
        }.get(role, "Most frequent imputation")

        rows.append({
            "column": column,
            "missing_count": missing_count,
            "missing_ratio": round(
                missing_ratio,
                3
            ),
            "strategy": strategy,
            "role": role,
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
    text_columns=None,
    one_hot_drop=None,
    boolean_columns=None,
    date_columns=None,
    text_max_features=5000,
):
    """
    Build a scikit-learn ColumnTransformer.

    The transformer must only be FITTED on training data.
    The test fold is transformed using the already-fitted
    transformer. This prevents data leakage.
    """

    text_columns = text_columns or []
    boolean_columns = boolean_columns or []
    date_columns = date_columns or []

    transformers = []

    # ---------------- NUMERIC ----------------

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

    # ---------------- CATEGORICAL ----------------

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

    # ---------------- BOOLEAN ----------------

    if boolean_columns:

        boolean_pipeline = Pipeline(
            steps=[
                (
                    "as_object",
                    FunctionTransformer(
                        _as_object,
                        feature_names_out="one-to-one"
                    )
                ),
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "to_numeric",
                    FunctionTransformer(
                        _boolean_to_int,
                        feature_names_out="one-to-one"
                    )
                )
            ]
        )

        transformers.append(
            (
                "boolean",
                boolean_pipeline,
                boolean_columns
            )
        )

    # ---------------- TEXT (one transformer per column) ----------------

    for column in text_columns:

        text_pipeline = Pipeline(
            steps=[
                (
                    "to_text",
                    FunctionTransformer(
                        _flatten_text,
                        feature_names_out="one-to-one"
                    )
                ),
                (
                    "tfidf",
                    _PrefixedTfidf(
                        column=column,
                        max_features=text_max_features
                    )
                )
            ]
        )

        transformers.append(
            (
                f"text_{column}",
                text_pipeline,
                [column]
            )
        )

    # ---------------- DATETIME ----------------

    if date_columns:

        date_transformer = _DateComponents()

        transformers.append(
            (
                "datetime",
                date_transformer,
                date_columns
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )


def _boolean_to_int(X):
    """
    Convert boolean / 0-1 values to integers 0/1.
    """
    arr = np.asarray(X)

    if arr.dtype == object or arr.dtype == bool:
        return arr.astype(int)

    return arr


def _as_object(X):
    """
    Cast a column to object dtype so SimpleImputer can
    handle boolean values.
    """
    return np.asarray(X, dtype=object)


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
            # Stringify keys so target_info stays JSON
            # serializable even for numeric labels.
            # The raw mapping above is still used for
            # encoding; only the reported copy changes.
            "mapping": {
                str(label): index
                for label, index
                in mapping.items()
            },
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
    feature_columns
):
    """
    Fit the pipeline on the full feature matrix to show
    what the transformed data looks like.

    NOTE: This is only a preview for display purposes.
    During real training the pipeline is fitted on the
    training fold only, never on the full dataset.
    """

    if not feature_columns:
        return {
            "rows": int(len(df)),
            "features_before": 0,
            "features_after": 0,
            "feature_names": [],
            "is_sparse": False
        }

    X = df[feature_columns]

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", FutureWarning)
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
    problem_type="Classification",
    schema=None
):
    """
    Build the complete preprocessing plan for the dataset.

    This does NOT fit anything on the full dataset for
    real training. It returns the pipeline and a report
    that the UI can display.
    """

    if problem_type == "Clustering":
        target_column = None

    # Modelling rows exclude missing-target records. The
    # original frame is preserved for reporting; the
    # preview and counts below use the cleaned frame so
    # row counts stay consistent with training.
    df_model, row_info = drop_missing_target_rows(
        df, target_column
    )

    if schema is None:
        schema = analyze_schema(
            df,
            target_column=target_column,
        )

    numerical_columns = schema["numerical_columns"]
    categorical_columns = schema["categorical_columns"]
    boolean_columns = schema["boolean_columns"]
    text_columns = schema["text_columns"]
    date_columns = schema["date_columns"]

    pipeline = build_preprocessing_pipeline(
        numerical_columns,
        categorical_columns,
        text_columns,
        boolean_columns=boolean_columns,
        date_columns=date_columns,
    )

    missing_summary = missing_value_summary(
        df_model,
        schema["feature_columns"],
        schema=schema,
    )

    target, target_info = prepare_target(
        df_model,
        target_column,
        problem_type
    )

    preview = preview_transformation(
        df_model,
        pipeline,
        schema["feature_columns"],
    )

    _expansion_estimate, _expansion_warning = (
        _categorical_expansion_estimate(preview, schema)
    )
    preview["categorical_expansion_estimate"] = _expansion_estimate
    preview["expansion_warning"] = _expansion_warning

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

    if boolean_columns:
        steps.append(
            f"Boolean features ({len(boolean_columns)}): "
            "most-frequent imputation + binary conversion"
        )

    if text_columns:
        steps.append(
            f"Text features ({len(text_columns)}): "
            "TF-IDF vectorization (unigrams + bigrams, "
            "max 5000 terms per column)"
        )

    if date_columns:
        steps.append(
            f"Date features ({len(date_columns)}): "
            "year / month / day / day-of-week / hour "
            "extraction"
        )

    if schema["id_columns"]:
        steps.append(
            f"Identifier columns removed "
            f"({len(schema['id_columns'])}): "
            f"{', '.join(schema['id_columns'])}"
        )

    if schema["constant_columns"]:
        steps.append(
            f"Constant columns removed "
            f"({len(schema['constant_columns'])}): "
            f"{', '.join(schema['constant_columns'])}"
        )

    if schema["unsupported_columns"]:
        steps.append(
            f"Unsupported columns excluded "
            f"({len(schema['unsupported_columns'])}): "
            f"{', '.join(schema['unsupported_columns'])}"
        )

    if target_column:
        steps.append(
            "Target handled separately; missing target rows "
            "excluded; target encoded for modelling"
        )

        if row_info["rows_excluded_missing_target"]:
            steps.append(row_info["target_warning"])

    else:
        steps.append(
            "Clustering mode: all prepared columns are "
            "used as features"
        )

    return {
        "target_column": target_column,
        "schema": schema,
        "roles": schema["by_role"],
        "pipeline": pipeline,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "boolean_columns": boolean_columns,
        "text_columns": text_columns,
        "date_columns": date_columns,
        "unsupported_columns": schema["unsupported_columns"],
        "id_columns": schema["id_columns"],
        "constant_columns": schema["constant_columns"],
        "missing_summary": missing_summary,
        "target_info": target_info,
        "preview": preview,
        "steps": steps,
        "n_rows": row_info["rows_available_for_modeling"],
        "rows_uploaded": row_info["rows_uploaded"],
        "row_info": row_info,
        "has_missing": any(
            item["missing_count"] > 0
            for item in missing_summary
        ),
        "feature_accounting": feature_accounting(
            df,
            target_column,
            schema,
            preview,
            row_info=row_info,
        ),
    }


EXPANSION_WARNING_THRESHOLD = 500


def _categorical_expansion_estimate(preview, schema):
    """
    Estimate how many transformed features come from
    categorical one-hot expansion (total minus TF-IDF
    terms minus 1:1 numeric/boolean and 5-per-date
    features). Returns (estimate, warning_or_None).
    """
    if not preview:
        return 0, None

    total = int(preview.get("features_after", 0))
    names = preview.get("feature_names", []) or []
    text_columns = set(schema.get("text_columns", []))

    tfidf_terms = sum(
        1 for name in names
        if ":" in str(name)
        and str(name).split(":", 1)[0] in text_columns
    )

    estimate = (
        total
        - tfidf_terms
        - len(schema.get("numerical_columns", []))
        - len(schema.get("boolean_columns", []))
        - 5 * len(schema.get("date_columns", []))
    )
    estimate = max(0, int(estimate))

    if estimate >= EXPANSION_WARNING_THRESHOLD:
        return estimate, (
            f"Categorical expansion created about "
            f"{estimate} transformed features. One-hot "
            f"encoding of high-cardinality columns inflates "
            f"the feature matrix; consider rare-category "
            f"grouping or frequency encoding."
        )

    return estimate, None


def feature_accounting(
    df,
    target_column,
    schema,
    preview,
    row_info=None,
):
    """
    Consistent feature-count accounting shared by
    preprocessing, training, explainability and reports.

    TaskAware does not automatically drop features found
    by the advisory selection report, so selection is
    reported as not applied and every transformed feature
    is retained for modelling.
    """
    usable = list(schema.get("feature_columns", []))
    transformed = int(
        (preview or {}).get("features_after", 0)
    )

    if row_info is None:
        row_info = {
            "rows_uploaded": int(len(df)),
            "rows_available_for_modeling": int(len(df)),
            "rows_excluded_missing_target": 0,
            "target_missing_count": 0,
            "target_warning": None,
        }

    expansion_estimate, expansion_warning = (
        _categorical_expansion_estimate(preview, schema)
    )

    return {
        "raw_columns_total": int(len(df.columns)),
        "target_column": target_column,
        "raw_feature_columns": (
            int(len(df.columns) - 1)
            if target_column in list(df.columns)
            else int(len(df.columns))
        ),
        "identifier_columns_removed": list(
            schema.get("id_columns", [])
        ),
        "unsupported_columns_excluded": list(
            schema.get("unsupported_columns", [])
        ),
        "constant_columns_removed": list(
            schema.get("constant_columns", [])
        ),
        "raw_usable_feature_columns": usable,
        "raw_usable_count": len(usable),
        "generated_transformed_features": transformed,
        "feature_selection_applied": False,
        "selection_note": (
            "Feature selection was not applied. All "
            "transformed features were retained."
        ),
        "features_removed_by_selection": 0,
        "final_model_features": transformed,
        "rows_uploaded": row_info["rows_uploaded"],
        "rows_available_for_modeling": row_info[
            "rows_available_for_modeling"
        ],
        "rows_excluded_missing_target": row_info[
            "rows_excluded_missing_target"
        ],
        "target_missing_count": row_info["target_missing_count"],
        "target_warning": row_info["target_warning"],
        "categorical_expansion_estimate": expansion_estimate,
        "expansion_warning": expansion_warning,
    }