"""
Central schema analyzer for TaskAware.

Every module that needs to know the role of each column must
use analyze_schema() instead of guessing column types itself.

Roles:
    numeric       - continuous or integer-valued columns
    categorical   - discrete labels (e.g. gender, region)
    boolean       - True/False columns
    text          - natural-language free text (reviews, comments)
    datetime      - date / datetime columns
    identifier    - keys, IDs, UUIDs, machine-generated codes
    unsupported   - images, nested objects, lists, dicts, etc.
    constant      - columns with a single value (no information)

Detection rules
---------------
Text detection runs BEFORE identifier detection. A column that
contains natural-language text is classified as text even when
almost every value is unique.

Identifier detection requires more than a high unique ratio.
It combines:
    - column name keywords (id, uuid, key, code, ...)
    - unique ratio
    - average value length
    - number of words
    - whether values look like keys / codes
    - whether values look like natural-language sentences
"""

import re

import numpy as np
import pandas as pd


ROLE_NUMERIC = "numeric"
ROLE_CATEGORICAL = "categorical"
ROLE_BOOLEAN = "boolean"
ROLE_TEXT = "text"
ROLE_DATETIME = "datetime"
ROLE_IDENTIFIER = "identifier"
ROLE_UNSUPPORTED = "unsupported"
ROLE_CONSTANT = "constant"

# ------------------------------------------------------------
# VALUE SHAPE HELPERS
# ------------------------------------------------------------

_ID_NAME_TOKENS = {
    "id",
    "ids",
    "identifier",
    "identifiers",
    "uuid",
    "uuid4",
    "key",
    "keys",
    "code",
    "codes",
    "hash",
    "token",
    "guid",
    "serial",
    "serialno",
    "serialnumber",
    "rollno",
    "rollnumber",
    "registrationid",
    "accountnumber",
    "transactionid",
    "customerid",
    "userid",
    "studentid",
    "recordid",
    "productid",
    "orderid",
    "sessionid",
    "deviceid",
    "paymentid",
    "jobid",
    "subscriptionid",
    "invoiceid",
}

_TEXT_NAME_TOKENS = {
    "review",
    "reviews",
    "comment",
    "comments",
    "description",
    "descriptions",
    "feedback",
    "message",
    "messages",
    "article",
    "articles",
    "content",
    "body",
    "text",
    "tweet",
    "tweets",
    "abstract",
    "summary",
    "paragraph",
    "transcript",
    "headline",
    "caption",
    "essay",
    "story",
    "note",
    "notes",
    "critic",
    "opinion",
    "sentence",
    "reviewbody",
}

# Patterns that indicate a machine-generated / structured code
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_CODE_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

_NUMERIC_CODE_RE = re.compile(r"^\d+$")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)


def _identifier_by_name_only(column_name):
    """
    Whether the column name alone suggests an identifier.

    Used for columns whose name contains id-like keywords,
    applied BEFORE type-based detection so that e.g.
    integer 'student_id' columns are correctly treated as
    identifiers.
    """
    tokens = _normalize_name(column_name).split()

    text_tokens = {
        "text",
        "review",
        "comment",
        "description",
        "feedback",
        "message",
        "article",
        "content",
        "body",
        "essay",
        "note",
    }

    id_hit = any(
        token in _ID_NAME_TOKENS
        for token in tokens
    )

    text_hit = any(
        token in text_tokens
        for token in tokens
    )

    # "review_id" is an id; "review" alone is text.
    return id_hit and not text_hit


def _normalize_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def _value_lengths(series, limit=200):
    """
    Return average/max/min character lengths and average word
    counts for a sampled subset of non-null string values.
    """
    valid = series.dropna()

    if len(valid) == 0:
        return {
            "avg_length": 0.0,
            "max_length": 0,
            "min_length": 0,
            "avg_words": 0.0,
        }

    sample = valid.sample(
        n=min(limit, len(valid)),
        random_state=42,
    )

    lengths = sample.astype(str).str.len()

    word_counts = (
        sample.astype(str)
        .str.split()
        .str.len()
    )

    return {
        "avg_length": round(
            float(lengths.mean()),
            2
        ),
        "max_length": int(lengths.max()),
        "min_length": int(lengths.min()),
        "avg_words": round(
            float(word_counts.mean()),
            2
        ),
    }


def _word_count_series(series, limit=200):
    valid = series.dropna()

    if len(valid) == 0:
        return pd.Series(dtype=float)

    sample = valid.sample(
        n=min(limit, len(valid)),
        random_state=42,
    )

    return sample.astype(str).str.split().str.len()


def _has_natural_language(series, limit=200, column_name=None):
    """
    Determine whether a column contains natural-language text.

    Evidence used:
    - average word count >= 4
    - a large share of values have several words
    - average character length is substantial
    - column name suggests free text (supporting signal)
    """
    if pd.api.types.is_numeric_dtype(series):
        return False

    valid = series.dropna().astype(str)

    if len(valid) == 0:
        return False

    sample = valid.sample(
        n=min(limit, len(valid)),
        random_state=42,
    )

    words = sample.str.split().str.len()

    multiword_ratio = float(
        (words >= 5).mean()
    )

    avg_words = float(words.mean())

    avg_length = float(sample.str.len().mean())

    # Column name is a supporting signal for text
    name_suggests_text = False

    if column_name:
        tokens = _normalize_name(column_name).split()
        name_suggests_text = any(
            token in _TEXT_NAME_TOKENS
            for token in tokens
        )

    if avg_words >= 8 and multiword_ratio >= 0.5:
        return True

    if avg_length >= 60 and avg_words >= 6:
        return True

    if avg_words >= 4 and multiword_ratio >= 0.8:
        return True

    if (
        name_suggests_text
        and avg_words >= 3
        and multiword_ratio >= 0.3
    ):
        return True

    # A column explicitly named as text (review, comment,
    # description, ...) with several words per value is
    # treated as text even when values are short.
    if name_suggests_text and avg_words >= 3:
        return True

    return False


def _looks_like_identifier(series, column_name, unique_ratio):
    """
    Multi-signal identifier detection.

    Returns (is_identifier, list_of_reasons).
    """
    tokens = _normalize_name(column_name).split()

    name_suggests_id = any(
        token in _ID_NAME_TOKENS
        for token in tokens
    )

    name_suggests_text = any(
        token in _TEXT_NAME_TOKENS
        for token in tokens
    )

    reasons = []

    if name_suggests_id:
        reasons.append(
            "The column name contains identifier keywords "
            f"(e.g. '{column_name}')."
        )

    if name_suggests_id and not name_suggests_text:
        return True, reasons

    is_numeric = pd.api.types.is_numeric_dtype(series)

    valid = series.dropna().astype(str)

    if len(valid) == 0:
        return False, reasons

    sample = valid.sample(
        n=min(200, len(valid)),
        random_state=42,
    )

    lengths = sample.str.len()

    words = sample.str.split().str.len()

    avg_words = float(words.mean())

    avg_length = float(lengths.mean())

    # Natural-language text is never treated as an ID.
    if avg_words >= 4:
        return False, reasons

    # ---------------- NUMERIC COLUMNS ----------------
    if is_numeric:

        numeric = pd.to_numeric(
            series,
            errors="coerce"
        ).dropna()

        if len(numeric) == 0:
            return False, reasons

        n = len(numeric)

        is_sequential = bool(
            np.array_equal(
                np.sort(numeric.unique()),
                np.arange(
                    numeric.min(),
                    numeric.max() + 1,
                    dtype="float64",
                ),
            )
        )

        if unique_ratio >= 0.999 and n >= 10 and is_sequential:
            reasons.append(
                "The column contains unique sequential "
                "integer values, typical of a row index "
                "or generated ID."
            )
            return True, reasons

        # Continuous numeric features (income, age, price)
        # are never IDs even when every value is unique.
        return False, reasons

    # ---------------- STRING COLUMNS ----------------

    # Code-like values (no spaces, all alnum/hyphen/underscore)
    code_ratio = float(
        sample.apply(
            lambda v: bool(
                _CODE_RE.match(str(v).strip())
            )
        ).mean()
    )

    uuid_ratio = float(
        sample.apply(
            lambda v: bool(
                _UUID_RE.match(str(v).strip())
            )
        ).mean()
    )

    email_ratio = float(
        sample.apply(
            lambda v: bool(
                _EMAIL_RE.match(str(v).strip())
            )
        ).mean()
    )

    url_ratio = float(
        sample.apply(
            lambda v: bool(
                _URL_RE.match(str(v).strip())
            )
        ).mean()
    )

    if unique_ratio >= 0.95:

        # Almost every value unique is a strong signal,
        # but only when values look like codes / keys and
        # NOT like natural language.
        if uuid_ratio >= 0.9:
            reasons.append(
                "Almost every value is unique and values "
                "look like UUIDs."
            )
            return True, reasons

        if (
            code_ratio >= 0.9
            and avg_words <= 2
            and not email_ratio >= 0.9
            and not url_ratio >= 0.9
        ):
            reasons.append(
                "Almost every value is unique and values "
                "look like structured codes or keys."
            )
            return True, reasons

    if unique_ratio >= 0.999 and avg_words <= 1:

        # Emails and URLs are identifiers-like keys but
        # remain usable features, so do not remove them.
        if not email_ratio >= 0.9 and not url_ratio >= 0.9:
            reasons.append(
                "Every value is unique and values are short "
                "single tokens."
            )
            return True, reasons

    return False, reasons


# ------------------------------------------------------------
# UNSUPPORTED TYPE DETECTION
# ------------------------------------------------------------

def _is_unsupported(series):
    """
    Detect columns that TaskAware cannot use as features:
    nested objects, lists, dicts, images, raw bytes, etc.
    """
    dtype = series.dtype

    if isinstance(dtype, np.dtype):
        if dtype.kind == "c":
            return True

        if dtype.kind in ("O",):
            sample = series.dropna().head(50)

            if len(sample) == 0:
                return False

            for value in sample:
                if isinstance(
                    value,
                    (dict, list, tuple, set, bytes, bytearray),
                ):
                    return True

    return False


def _looks_like_datetime(series):
    """
    Detect datetime columns, including string columns that
    parse cleanly as dates.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    valid = series.dropna().astype(str)

    if len(valid) == 0:
        return False

    sample = valid.sample(
        n=min(100, len(valid)),
        random_state=42,
    )

    parsed = pd.to_datetime(
        sample,
        errors="coerce",
        format="mixed",
    )

    parse_ratio = float(parsed.notna().mean())

    return parse_ratio >= 0.9


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------

def analyze_schema(df, target_column=None):
    """
    Classify every column in the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
    target_column : str or None

    Returns
    -------
    dict
        {
          "columns": [ per-column role records ],
          "by_role": { role: [column names] },
          "target_column": str or None,
          "target_role": str or None,
          "constant_columns": [...],
          "id_columns": [...],
          "unsupported_columns": [...],
          "text_columns": [...],
          "date_columns": [...],
          "numerical_columns": [...],
          "categorical_columns": [...],
          "boolean_columns": [...],
          "feature_columns": [...],
          "removed_columns": [...],
          "is_time_series": bool,
          "time_column": str or None,
        }
    """

    columns = []

    by_role = {
        ROLE_NUMERIC: [],
        ROLE_CATEGORICAL: [],
        ROLE_BOOLEAN: [],
        ROLE_TEXT: [],
        ROLE_DATETIME: [],
        ROLE_IDENTIFIER: [],
        ROLE_UNSUPPORTED: [],
        ROLE_CONSTANT: [],
    }

    time_column = None

    for column in df.columns:

        series = df[column]

        # ------------------------------------------------
        # UNSUPPORTED (before nunique, which fails on
        # unhashable objects)
        # ------------------------------------------------

        if _is_unsupported(series):

            role = ROLE_UNSUPPORTED

            reasons = [
                "The column contains unsupported object "
                "types (lists, dicts, bytes or similar)."
            ]

            unique_count = 0
            missing_count = int(series.isnull().sum())
            missing_ratio = (
                missing_count / len(df)
                if len(df) > 0
                else 0.0
            )
            unique_ratio = 0.0
            shape = None

            is_target = (column == target_column)

            columns.append({
                "column": str(column),
                "role": role,
                "dtype": str(series.dtype),
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
                "missing_count": missing_count,
                "missing_ratio": round(missing_ratio, 3),
                "is_target": is_target,
                "reasons": reasons,
                "shape": shape,
            })

            by_role[role].append(column)

            continue

        unique_count = int(
            series.nunique(dropna=True)
        )

        missing_count = int(
            series.isnull().sum()
        )

        missing_ratio = (
            missing_count / len(df)
            if len(df) > 0
            else 0.0
        )

        unique_ratio = (
            unique_count / len(df)
            if len(df) > 0
            else 0.0
        )

        is_target = (
            column == target_column
        )

        # ------------------------------------------------
        # CONSTANT
        # ------------------------------------------------

        if unique_count <= 1:

            role = ROLE_CONSTANT

            reasons = [
                "The column contains a single constant "
                "value and carries no predictive "
                "information."
            ]

        # ------------------------------------------------
        # NAME-BASED IDENTIFIER
        #
        # A column whose name contains identifier keywords
        # (id, uuid, key, code, ...) is treated as an ID
        # regardless of dtype, unless it is clearly text.
        # ------------------------------------------------

        elif _looks_like_identifier(
            series,
            column,
            unique_ratio,
        )[0] and _identifier_by_name_only(column):

            role = ROLE_IDENTIFIER

            reasons = _looks_like_identifier(
                series,
                column,
                unique_ratio,
            )[1]

        # ------------------------------------------------
        # DATETIME
        # ------------------------------------------------

        elif _looks_like_datetime(series):

            role = ROLE_DATETIME

            reasons = [
                "The column is detected as a date or "
                "datetime column."
            ]

            if time_column is None:
                time_column = column

        # ------------------------------------------------
        # BOOLEAN
        # ------------------------------------------------

        elif pd.api.types.is_bool_dtype(series):

            role = ROLE_BOOLEAN

            reasons = [
                "The column contains boolean True/False "
                "values."
            ]

        # ------------------------------------------------
        # NUMERIC
        # ------------------------------------------------

        elif pd.api.types.is_numeric_dtype(series):

            # Integer columns that are fully unique and
            # look like sequential codes are treated as IDs
            # even without an id-like name.
            is_id, id_reasons = _looks_like_identifier(
                series,
                column,
                unique_ratio,
            )

            if is_id:

                role = ROLE_IDENTIFIER

                reasons = id_reasons

            elif unique_count <= 10:
                # Low-cardinality integers (e.g. 0/1, 1-5)
                # behave like categorical labels.
                role = ROLE_CATEGORICAL

                reasons = [
                    "The column is numeric but contains "
                    "few distinct values and behaves like "
                    "a categorical variable."
                ]

            else:

                role = ROLE_NUMERIC

                reasons = [
                    "The column contains numerical values."
                ]

        # ------------------------------------------------
        # TEXT (detected BEFORE identifier)
        # ------------------------------------------------

        elif _has_natural_language(
            series,
            column_name=column,
        ):

            role = ROLE_TEXT

            reasons = [
                "The column contains natural-language text "
                "and will be processed with TF-IDF."
            ]

        # ------------------------------------------------
        # IDENTIFIER (non-numeric)
        # ------------------------------------------------

        else:

            is_id, id_reasons = _looks_like_identifier(
                series,
                column,
                unique_ratio,
            )

            if is_id:

                role = ROLE_IDENTIFIER

                reasons = id_reasons

            else:

                role = ROLE_CATEGORICAL

                reasons = [
                    "The column contains discrete "
                    "categorical values."
                ]

        # ------------------------------------------------
        # RECORD
        # ------------------------------------------------

        shape = (
            _value_lengths(series)
            if role in (ROLE_TEXT, ROLE_IDENTIFIER, ROLE_CATEGORICAL)
            else None
        )

        columns.append({
            "column": str(column),
            "role": role,
            "dtype": str(series.dtype),
            "unique_count": unique_count,
            "unique_ratio": round(unique_ratio, 3),
            "missing_count": missing_count,
            "missing_ratio": round(missing_ratio, 3),
            "is_target": is_target,
            "reasons": reasons,
            "shape": shape,
        })

        by_role[role].append(column)

    # ------------------------------------------------
    # AGGREGATED LISTS
    #
    # Feature role lists must EXCLUDE the target column so
    # that the target is never treated as a feature.
    # ------------------------------------------------

    def _exclude_target(cols):
        return [
            col
            for col in cols
            if col != target_column
        ]

    id_columns = _exclude_target(
        by_role[ROLE_IDENTIFIER]
    )
    unsupported_columns = _exclude_target(
        by_role[ROLE_UNSUPPORTED]
    )
    constant_columns = _exclude_target(
        by_role[ROLE_CONSTANT]
    )
    text_columns = _exclude_target(
        by_role[ROLE_TEXT]
    )
    date_columns = _exclude_target(
        by_role[ROLE_DATETIME]
    )
    numerical_columns = _exclude_target(
        by_role[ROLE_NUMERIC]
    )
    categorical_columns = _exclude_target(
        by_role[ROLE_CATEGORICAL]
    )
    boolean_columns = _exclude_target(
        by_role[ROLE_BOOLEAN]
    )

    removed_columns = (
        id_columns
        + unsupported_columns
        + constant_columns
    )

    feature_columns = [
        col
        for col in df.columns
        if col != target_column
        and col not in removed_columns
    ]

    return {
        "columns": columns,
        "by_role": by_role,
        "target_column": (
            target_column
            if target_column in df.columns
            else None
        ),
        "target_role": next(
            (
                record["role"]
                for record in columns
                if record["column"] == target_column
            ),
            None,
        ),
        "constant_columns": constant_columns,
        "id_columns": id_columns,
        "unsupported_columns": unsupported_columns,
        "text_columns": text_columns,
        "date_columns": date_columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "boolean_columns": boolean_columns,
        "feature_columns": feature_columns,
        "removed_columns": removed_columns,
        "is_time_series": time_column is not None,
        "time_column": time_column,
    }


def get_role(schema, column):
    """
    Convenience accessor for a single column role.
    """
    for record in schema["columns"]:
        if record["column"] == column:
            return record["role"]
    return None