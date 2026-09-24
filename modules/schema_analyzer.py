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
# BOOLEAN DETECTION
# ------------------------------------------------------------

_BOOLEAN_TRUE_SET = {"true", "t", "yes", "y", "1"}
_BOOLEAN_FALSE_SET = {"false", "f", "no", "n", "0"}
_BOOLEAN_ENCODINGS = {
    "true_false": {"true", "false"},
    "yes_no": {"yes", "no"},
    "y_n": {"y", "n"},
    "t_f": {"t", "f"},
}


def _looks_like_boolean(series):
    """
    Detect Boolean-like columns.

    Returns (boolean_likelihood, encoding, evidence, warnings).
    """
    if pd.api.types.is_bool_dtype(series):
        return 1.0, "bool", [
            "The column uses native Boolean dtype."
        ], []

    valid = series.dropna()

    if len(valid) == 0:
        return 0.0, None, [], []

    # Numeric 0/1
    if pd.api.types.is_numeric_dtype(series):
        unique = set(valid.astype(int).unique()) if (
            (valid.astype(int) == valid).all()
        ) else set()
        if unique <= {0, 1} and len(unique) >= 1:
            return 0.9, "0_1", [
                "The column contains only 0/1 values."
            ], [
                "Numeric 0/1 values may be a Boolean feature, "
                "a binary classification target, or a numerical "
                "indicator. TaskAware keeps the raw role and "
                "reports the Boolean likelihood."
            ]
        return 0.0, None, [], []

    # String values
    sample = valid.astype(str).str.strip().str.lower()

    if len(sample) == 0:
        return 0.0, None, [], []

    unique_vals = set(sample.unique())

    for encoding, allowed in _BOOLEAN_ENCODINGS.items():

        if unique_vals and unique_vals <= allowed and len(unique_vals) >= 1:
            return 0.9, encoding, [
                f"The column uses Boolean-style values "
                f"({', '.join(sorted(unique_vals))})."
            ], []

    # Mixed-case True/False etc. covered above via lower()
    # If it contains both true-ish and false-ish tokens
    if unique_vals and (
        unique_vals <= _BOOLEAN_TRUE_SET
        or unique_vals <= _BOOLEAN_FALSE_SET
    ):
        return 0.9, "binary_string", [
            "The column contains a single Boolean-like token set."
        ], []

    return 0.0, None, [], []


# ------------------------------------------------------------
# SEMANTIC TYPE DETECTION (NUMERIC)
# ------------------------------------------------------------

_NUMERIC_MEASUREMENT_TOKENS = {
    "price", "cost", "amount", "fee", "income", "salary",
    "temperature", "weight", "height", "age", "score", "hours",
    "time", "length", "width", "depth", "distance", "speed",
    "rate", "value", "measurement",
}

_NUMERIC_COUNT_TOKENS = {
    "count", "number", "total", "quantity", "qty", "num",
    "n_", "items", "visits", "transactions", "purchases",
    "orders", "login", "clicks", "views", "members", "profiles",
    "people", "children", "products",
}

_NUMERIC_RATIO_TOKENS = {
    "ratio", "percentage", "percent", "proportion", "share",
    "conversion", "rate", "fraction", "pct",
}


def _detect_numeric_semantic(series, column_name, unique_count):
    """
    Determine the semantic type of a numeric column.

    Returns
    -------
    (semantic_type, likelihood_dict, confidence, evidence)

    likelihood_dict maps possible semantic types to
    probabilities summing to ~1.0.
    """
    name = _normalize_name(column_name)

    tokens = name.split()

    valid = series.dropna()

    is_integer = pd.api.types.is_integer_dtype(series) or (
        pd.api.types.is_float_dtype(series)
        and bool((valid.astype(int) == valid).all())
    )

    non_negative = bool((valid >= 0).all()) if len(valid) else False

    in_unit_range = bool(
        (valid >= 0).all() and (valid <= 1).all()
    ) if len(valid) else False

    evidence = []

    # Ratio / percentage
    name_ratio = any(
        token in _NUMERIC_RATIO_TOKENS
        for token in tokens
    )

    if name_ratio or in_unit_range:

        if in_unit_range and not name_ratio:
            evidence.append(
                "All values lie in the [0, 1] unit range, "
                "typical of a ratio."
            )
        if name_ratio:
            evidence.append(
                "The column name suggests a ratio or "
                "percentage."
            )

        return "ratio", {
            "measurement": 0.10,
            "count": 0.05,
            "ratio": 0.85,
        }, 0.85, evidence

    # Count by explicit count-like name
    name_count = any(
        token in _NUMERIC_COUNT_TOKENS
        for token in tokens
    )

    name_measure = any(
        token in _NUMERIC_MEASUREMENT_TOKENS
        for token in tokens
    )

    if is_integer and non_negative and name_count:

        evidence.append(
            "Integer, non-negative values with a "
            "count-like column name."
        )

        return "count", {
            "measurement": 0.10,
            "count": 0.85,
            "ratio": 0.05,
        }, 0.85, evidence

    # Measurement by explicit measurement-like name.
    # Checked BEFORE the generic low-cardinality count
    # heuristic so that integer-valued measurements
    # (e.g. age in whole years) are not misclassified
    # as counts.
    if name_measure and not name_count:
        evidence.append(
            "The column name suggests a continuous "
            "measurement."
        )

        return "measurement", {
            "measurement": 0.80,
            "count": 0.15,
            "ratio": 0.05,
        }, 0.80, evidence

    if (
        is_integer
        and non_negative
        and unique_count < 100
        and not name_measure
    ):

        evidence.append(
            "Integer, non-negative values with limited "
            "cardinality, typical of a count."
        )

        return "count", {
            "measurement": 0.30,
            "count": 0.60,
            "ratio": 0.10,
        }, 0.60, evidence

    # Measurement (default)
    if name_measure:
        evidence.append(
            "The column name suggests a continuous "
            "measurement."
        )

    return "measurement", {
        "measurement": 0.70,
        "count": 0.20,
        "ratio": 0.10,
    }, 0.70, evidence


def _keep_numeric_role(series, column_name, unique_count):
    """
    Decide whether a low-cardinality numeric column should keep
    the numeric role instead of being treated as categorical.

    Binary 0/1-style columns stay categorical (they are reported
    as boolean-like / binary targets). Integer columns with a
    count-like name and more than two distinct values, and columns
    with a ratio-like name, keep the numeric role so their
    semantic type resolves to count / ratio.
    """
    boolean_likelihood, _, _, _ = _looks_like_boolean(series)

    if boolean_likelihood >= 0.9 and unique_count <= 2:
        return False

    tokens = _normalize_name(column_name).split()

    name_count = any(
        token in _NUMERIC_COUNT_TOKENS
        for token in tokens
    )

    name_ratio = any(
        token in _NUMERIC_RATIO_TOKENS
        for token in tokens
    )

    if name_ratio:
        return True

    if name_count and unique_count > 2:
        valid = series.dropna()

        if len(valid) == 0:
            return False

        try:
            is_integer = bool(
                (valid.astype(int) == valid).all()
            )
            non_negative = bool((valid >= 0).all())
        except Exception:
            return False

        return bool(is_integer and non_negative)

    return False


# ------------------------------------------------------------
# CARDINALITY LEVELS
# ------------------------------------------------------------

def _cardinality_level(unique_count, unique_ratio, n_rows):
    """
    Classify categorical cardinality into low / medium /
    high / extreme considering both absolute count and ratio.
    """
    if n_rows == 0:
        return "low", None

    if unique_count <= 10:
        return "low", None

    if unique_count <= 50 or unique_ratio <= 0.05:
        return "medium", None

    if unique_count <= 500 or unique_ratio <= 0.2:
        return "high", (
            "High-cardinality categorical column. One-hot "
            "encoding would create many columns; consider "
            "rare-category grouping, frequency encoding, or "
            "hashing."
        )

    return "extreme", (
        "Extreme-cardinality categorical column. One-hot "
        "encoding is not recommended; consider hashing, "
        "frequency encoding, or user confirmation before "
        "use."
    )


# ------------------------------------------------------------
# RECOMMENDED TRANSFORMATION
# ------------------------------------------------------------

def _recommended_transformation(record):
    """
    Produce the recommended transformation text for a column.
    """
    role = record["role"]
    semantic = record.get("semantic_type", role)
    cardinality = record.get("cardinality_level", "low")

    if role == "numeric":
        if semantic == "count":
            return (
                "Median imputation + scaling (or tree-"
                "compatible treatment)"
            )
        return "Median imputation + scaling"

    if role == "categorical":
        if cardinality in ("high", "extreme"):
            return (
                "Rare-category grouping / frequency encoding / "
                "hashing, or warning before one-hot encoding"
            )
        return "Most-frequent imputation + one-hot encoding"

    if role == "boolean":
        return "Boolean conversion + safe imputation"

    if role == "text":
        return "Missing-text handling + TF-IDF"

    if role == "datetime":
        return "Date-component extraction or time-aware processing"

    if role == "identifier":
        return "Exclude from predictive features"

    if role == "unsupported":
        return "Report details and require user choice"

    if role == "constant":
        return "Exclude (no information)"

    return "Target processed separately from features"


# ------------------------------------------------------------
# TARGET SUITABILITY
# ------------------------------------------------------------

def _target_suitability(record, unique_count, missing_ratio, n_rows):
    """
    Compute target suitability for a column.

    Returns a dict with technical/semantic suitability,
    leakage risk, target problem type, recommendation level,
    reasons and warnings.
    """
    role = record["role"]
    column = record["column"]

    reasons = []
    warnings = []
    recommendation_level = "unsuitable"
    technical_suitability = "low"
    semantic_suitability = "low"
    target_problem_type = None
    leakage_risk = "low"

    # Unusable roles
    if role in ("identifier", "unsupported", "constant"):
        reasons.append(
            f"The column is {role}; it cannot be used as a "
            "target."
        )
        if role == "identifier":
            reasons.append(
                "Identifiers are unique per row and carry no "
                "generalizable outcome."
            )
        return {
            "technical_suitability": "low",
            "semantic_suitability": "low",
            "objective_alignment": 0,
            "leakage_risk": leakage_risk,
            "confidence": "high",
            "target_problem_type": None,
            "reasons": reasons,
            "warnings": warnings,
            "recommendation_level": "excluded",
        }

    if role == "text":
        reasons.append(
            "Free-text targets are not supported by the "
            "current prediction engine."
        )
        return {
            "technical_suitability": "low",
            "semantic_suitability": "low",
            "objective_alignment": 0,
            "leakage_risk": "low",
            "confidence": "high",
            "target_problem_type": None,
            "reasons": reasons,
            "warnings": warnings,
            "recommendation_level": "excluded",
        }

    if missing_ratio > 0.5:
        warnings.append(
            "More than half of the values are missing."
        )
        technical_suitability = "low"
        recommendation_level = "weak"

    # Numeric 0/1 -> binary classification
    if role in ("categorical", "boolean", "numeric"):

        boolean_likelihood = record.get("boolean_likelihood", 0)

        if boolean_likelihood >= 0.9 and unique_count <= 2:

            target_problem_type = "Classification"
            technical_suitability = "high"
            semantic_suitability = "medium"
            recommendation_level = "recommended"

            reasons.append(
                "The column is a binary 0/1 or Boolean value; "
                "it is likely a binary classification target."
            )

            warnings.append(
                "The binary numerical target is treated as "
                "classification, not regression."
            )

        elif role in ("categorical", "boolean"):

            target_problem_type = "Classification"
            technical_suitability = "high"

            if unique_count <= 10:
                recommendation_level = "recommended"
                semantic_suitability = "medium"
                reasons.append(
                    f"The column has {unique_count} classes, "
                    "suitable for classification."
                )
            else:
                recommendation_level = "possible"
                semantic_suitability = "medium"
                warnings.append(
                    f"The column has {unique_count} classes, "
                    "which may make classification difficult."
                )

        elif role == "numeric":

            if unique_count >= 20:
                target_problem_type = "Regression"
                technical_suitability = "high"
                semantic_suitability = "medium"
                recommendation_level = "recommended"
                reasons.append(
                    "Continuous numerical values with good "
                    "variation are suitable for regression."
                )
            else:
                target_problem_type = "Classification"
                technical_suitability = "medium"
                semantic_suitability = "medium"
                recommendation_level = "possible"
                reasons.append(
                    "Low-cardinality numerical values may "
                    "represent class labels."
                )
                warnings.append(
                    "The numeric target has few distinct "
                    "values; interpret whether it is ordinal "
                    "or categorical."
                )

    # Leakage warning for names containing "target"/"label"
    # and strong correlation is handled elsewhere.

    if recommendation_level == "unsuitable" and role not in (
        "identifier", "unsupported", "constant", "text"
    ):
        recommendation_level = "possible"
        technical_suitability = "medium"

    return {
        "technical_suitability": technical_suitability,
        "semantic_suitability": semantic_suitability,
        "objective_alignment": 0,
        "leakage_risk": leakage_risk,
        "confidence": "moderate",
        "target_problem_type": target_problem_type,
        "reasons": reasons,
        "warnings": warnings,
        "recommendation_level": recommendation_level,
    }


# ------------------------------------------------------------
# GROUP DETECTION
# ------------------------------------------------------------

_GROUP_NAME_TOKENS = {
    "customer", "patient", "user", "account", "session",
    "household", "device", "transaction", "entity", "member",
    "employee", "subscriber", "student", "store", "branch",
    "order", "invoice",
}


def _detect_group(
    series,
    column_name,
    unique_count,
    unique_ratio,
    n_rows,
    role,
):
    """
    Detect repeated-entity / group columns.

    Returns a 7-tuple:
        (group_likelihood, group_candidate, group_confidence,
         repeated_entity_count, max_rows_per_entity,
         reason_list, warning_or_None)
    """
    tokens = _normalize_name(column_name).split()

    name_suggests_group = any(
        token in _GROUP_NAME_TOKENS
        for token in tokens
    )

    if unique_count == 0 or n_rows == 0:
        return 0.0, False, None, 0, 0, [], None

    value_counts = series.dropna().value_counts()

    repeated_entity_count = unique_count

    max_rows_per_entity = int(
        value_counts.max()
    ) if len(value_counts) else 0

    has_repeats = max_rows_per_entity >= 2

    # A unique ID with one row per entity is NOT a group.
    if unique_ratio >= 0.95:
        return 0.0, False, None, 0, 1, [], None

    reason = []
    warning = None

    if name_suggests_group and has_repeats:

        return (
            0.9,
            True,
            "high",
            repeated_entity_count,
            max_rows_per_entity,
            [
                "The column name suggests an entity and "
                "values repeat across rows."
            ],
            (
                "Repeated entity detected. Records from the "
                "same entity should not be split across "
                "training and validation without group-aware "
                "splitting."
            ),
        )

    if has_repeats and role == "identifier":

        return (
            0.8,
            True,
            "medium",
            repeated_entity_count,
            max_rows_per_entity,
            [
                "An identifier-like column with repeated "
                "values may identify a group/entity."
            ],
            (
                "This column is treated as an identifier for "
                "features, but its repeated values may "
                "represent groups."
            ),
        )

    if has_repeats and role in ("categorical", "numeric") and (
        unique_ratio < 0.3 and unique_ratio > 0.01
    ):

        return (
            0.5,
            True,
            "low",
            repeated_entity_count,
            max_rows_per_entity,
            [
                "Values repeat; the column may identify "
                "entities, but evidence is not strong."
            ],
            (
                "Possible group column with low confidence. "
                "Confirm whether records belong to repeated "
                "entities before group-aware splitting."
            ),
        )

    return 0.0, False, None, 0, 0, [], None


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


def _unsupported_details(series):
    """
    Return detailed information about an unsupported column.

    Returns a dict with detected types, reason, recommended
    action, impact and whether analysis can continue.
    """
    dtype = series.dtype

    detected_types = set()

    sample = series.dropna().head(50)

    for value in sample:

        if isinstance(value, dict):
            detected_types.add("dictionaries")
        elif isinstance(value, (list, tuple)):
            detected_types.add("lists/tuples")
        elif isinstance(value, set):
            detected_types.add("sets")
        elif isinstance(value, (bytes, bytearray)):
            detected_types.add("bytes")
        elif hasattr(value, "__array_interface__"):
            detected_types.add("raw embeddings/arrays")
        elif isinstance(value, (int, float, str, bool)):
            pass
        else:
            detected_types.add(type(value).__name__)

    if isinstance(dtype, np.dtype) and dtype.kind == "c":
        detected_types.add("complex numbers")

    type_label = (
        ", ".join(sorted(detected_types))
        if detected_types
        else str(dtype)
    )

    reason = (
        "The column contains values with a nested or "
        "complex structure that cannot be represented as "
        "flat tabular features."
    )

    return {
        "raw_dtype": str(dtype),
        "detected_value_types": type_label,
        "reason": reason,
        "recommended_action": (
            "Flatten the structure, transform the values "
            "into supported columns, or exclude the column."
        ),
        "impact": (
            "The column will not be used for modelling."
        ),
        "can_continue_without": True,
    }


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
# RECORD ENRICHMENT
# ------------------------------------------------------------

def _native(value):
    """
    Convert NumPy/Pandas values to native Python types so the
    schema record is JSON serializable.
    """
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (np.ndarray,)):
        return [_native(v) for v in value.tolist()]

    if isinstance(value, pd.Timestamp):
        return str(value)

    if isinstance(value, set):
        return sorted(value)

    if isinstance(value, dict):
        return {
            str(k): _native(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]

    if isinstance(value, (float, int, str, bool)):
        return value

    return str(value)


def _enrich_record(
    record,
    df,
    series,
    target_column,
):
    """
    Add the enriched schema fields to a base record.

    Preserves all existing keys and appends:
        raw_dtype, semantic_type, semantic_type_likelihood,
        confidence, text_likelihood, date_likelihood,
        identifier_likelihood, boolean_likelihood,
        boolean_encoding, cardinality_level,
        cardinality_warning, recommended_transformation,
        target_suitability, group_candidate, group_likelihood,
        group_confidence, repeated_entity_count,
        max_rows_per_entity, warnings, evidence,
        unsupported_details.
    """
    role = record["role"]
    column = record["column"]
    unique_count = record["unique_count"]
    unique_ratio = record["unique_ratio"]
    missing_count = record["missing_count"]
    missing_ratio = record["missing_ratio"]
    is_target = record["is_target"]
    n_rows = len(df)

    enriched = dict(record)

    enriched["raw_dtype"] = str(series.dtype)

    # ------------------------------------------------
    # LIKELIHOODS
    # ------------------------------------------------

    text_likelihood = 0.0
    date_likelihood = 0.0
    identifier_likelihood = 0.0

    if role == ROLE_TEXT:
        text_likelihood = 1.0
    elif role == ROLE_DATETIME:
        date_likelihood = 1.0
    elif role == ROLE_IDENTIFIER:
        identifier_likelihood = 1.0
    else:
        text_likelihood = (
            1.0 if _has_natural_language(series, column_name=column)
            else 0.0
        )
        date_likelihood = (
            1.0 if _looks_like_datetime(series)
            else 0.0
        )

        if _looks_like_identifier(series, column, unique_ratio)[0]:
            identifier_likelihood = 1.0

    boolean_likelihood, boolean_encoding, bool_evidence, bool_warnings = (
        _looks_like_boolean(series)
    )

    # ------------------------------------------------
    # SEMANTIC TYPE
    # ------------------------------------------------

    semantic_type = role
    semantic_likelihood = {}
    confidence = 1.0
    evidence = []

    if role == ROLE_NUMERIC:

        semantic_type, semantic_likelihood, confidence, evidence = (
            _detect_numeric_semantic(series, column, unique_count)
        )

    elif role == ROLE_CATEGORICAL:

        semantic_type = "categorical"

        if boolean_likelihood >= 0.9 and unique_count <= 2:
            semantic_type = "boolean"
            semantic_likelihood = {
                "categorical": 0.30,
                "boolean": 0.70,
            }
            confidence = 0.70
            evidence = bool_evidence
            enriched["warnings"] = bool_warnings
        else:
            semantic_likelihood = {
                "categorical": 0.90,
                "boolean": round(boolean_likelihood, 2),
            }
            confidence = 0.90

    elif role == ROLE_BOOLEAN:

        semantic_type = "boolean"
        semantic_likelihood = {
            "boolean": 1.0,
        }
        evidence = bool_evidence

    elif role == ROLE_TEXT:

        semantic_type = "text"
        semantic_likelihood = {"text": 1.0}
        evidence = [
            "The column contains natural-language text."
        ]

    elif role == ROLE_DATETIME:

        semantic_type = (
            "datetime"
            if "time" in str(series.dtype)
            else "date"
        )
        semantic_likelihood = {"date": 1.0}
        evidence = [
            "The column parses as a date or datetime."
        ]

    elif role == ROLE_IDENTIFIER:

        semantic_type = "identifier"
        semantic_likelihood = {"identifier": 1.0}
        evidence = list(record.get("reasons", []))

    elif role == ROLE_UNSUPPORTED:

        semantic_type = "unsupported"
        semantic_likelihood = {"unsupported": 1.0}

    elif role == ROLE_CONSTANT:

        semantic_type = "constant"
        semantic_likelihood = {"constant": 1.0}
        evidence = [
            "The column has a single unique value."
        ]

    # Target overrides semantic type
    if is_target:

        if role in (ROLE_CATEGORICAL, ROLE_BOOLEAN):

            if boolean_likelihood >= 0.9 and unique_count <= 2:
                semantic_type = "binary_target"
            else:
                semantic_type = "multiclass_target"

        elif role == ROLE_NUMERIC:

            if boolean_likelihood >= 0.9 and unique_count <= 2:
                semantic_type = "binary_target"
            elif unique_count <= 10:
                semantic_type = "multiclass_target"
            else:
                semantic_type = "regression_target"

    # ------------------------------------------------
    # CARDINALITY
    # ------------------------------------------------

    cardinality_level = "low"
    cardinality_warning = None

    if role in (ROLE_CATEGORICAL, ROLE_BOOLEAN):

        cardinality_level, cardinality_warning = (
            _cardinality_level(unique_count, unique_ratio, n_rows)
        )

    # ------------------------------------------------
    # RECOMMENDED TRANSFORMATION
    # ------------------------------------------------

    enriched["semantic_type"] = semantic_type
    enriched["semantic_type_likelihood"] = _native(
        semantic_likelihood
    )
    enriched["confidence"] = _native(confidence)
    enriched["text_likelihood"] = _native(text_likelihood)
    enriched["date_likelihood"] = _native(date_likelihood)
    enriched["identifier_likelihood"] = _native(
        identifier_likelihood
    )
    enriched["boolean_likelihood"] = _native(boolean_likelihood)
    enriched["boolean_encoding"] = (
        _native(boolean_encoding)
        if boolean_encoding
        else None
    )
    enriched["cardinality_level"] = cardinality_level
    enriched["cardinality_warning"] = (
        _native(cardinality_warning)
        if cardinality_warning
        else None
    )

    enriched["evidence"] = _native(evidence)

    enriched["recommended_transformation"] = (
        _recommended_transformation(enriched)
    )

    # ------------------------------------------------
    # TARGET SUITABILITY
    # ------------------------------------------------

    enriched["target_suitability"] = _native(
        _target_suitability(
            enriched,
            unique_count,
            missing_ratio,
            n_rows,
        )
    )

    # ------------------------------------------------
    # GROUP DETECTION
    # ------------------------------------------------

    # Boolean-like columns repeat only True/False-style values
    # and the target column repeats class labels; neither is a
    # repeated-entity group, so group detection is suppressed.
    suppress_group = bool(is_target) or boolean_likelihood >= 0.9

    if suppress_group:
        (
            group_likelihood,
            group_candidate,
            group_confidence,
            repeated_entity_count,
            max_rows_per_entity,
            group_reason,
            group_warning,
        ) = (0.0, False, None, 0, 0, [], None)
    else:
        (
            group_likelihood,
            group_candidate,
            group_confidence,
            repeated_entity_count,
            max_rows_per_entity,
            group_reason,
            group_warning,
        ) = _detect_group(
            series,
            column,
            unique_count,
            unique_ratio,
            n_rows,
            role,
        )

    enriched["group_likelihood"] = _native(group_likelihood)
    enriched["group_candidate"] = bool(group_candidate)
    enriched["group_confidence"] = group_confidence
    enriched["repeated_entity_count"] = _native(
        repeated_entity_count
    )
    enriched["max_rows_per_entity"] = _native(
        max_rows_per_entity
    )

    if group_reason:
        enriched["group_reason"] = _native(group_reason)
    else:
        enriched["group_reason"] = []

    if group_warning:
        enriched["warnings"] = enriched.get("warnings", []) + [
            _native(group_warning)
        ]

    # ------------------------------------------------
    # UNSUPPORTED DETAILS
    # ------------------------------------------------

    if role == ROLE_UNSUPPORTED:
        enriched["unsupported_details"] = _native(
            _unsupported_details(series)
        )
    else:
        enriched["unsupported_details"] = None

    # ------------------------------------------------
    # WARNINGS (merge existing)
    # ------------------------------------------------

    if "warnings" not in enriched:
        enriched["warnings"] = []

    return enriched


def _record_for(columns, column):
    """
    Return the schema record for a column, or an empty dict.
    """
    for record in columns:
        if record["column"] == column:
            return record
    return {}


def _validation_strategy(
    df,
    group_columns,
    time_column,
    target_column,
    columns,
):
    """
    Choose the validation strategy for the dataset based on
    group and time structure.
    """
    records = {
        record["column"]: record
        for record in columns
    }

    target_record = (
        records.get(target_column)
        if target_column in records
        else None
    )

    target_role = (
        target_record.get("role")
        if target_record
        else None
    )

    reason = []

    if time_column:
        reason.append(
            f"time-ordered data detected via "
            f"'{time_column}'"
        )

    if group_columns:
        reason.append(
            f"repeated entities detected in "
            f"{', '.join(group_columns)}"
        )

    if target_role in (ROLE_CATEGORICAL, ROLE_BOOLEAN):
        base_strategy = "StratifiedKFold"
    else:
        base_strategy = "KFold"

    if time_column and group_columns:
        strategy = (
            "Time-aware group-aware validation "
            "(grouped chronological split)"
        )
    elif group_columns:
        strategy = (
            "StratifiedGroupKFold"
            if target_role in (ROLE_CATEGORICAL, ROLE_BOOLEAN)
            else "GroupKFold"
        )
    elif time_column:
        strategy = (
            "TimeSeriesSplit / chronological split"
        )
    else:
        strategy = base_strategy

    return {
        "strategy": strategy,
        "group_columns": list(group_columns),
        "time_column": time_column,
        "number_of_folds": 5,
        "random_state": 42,
        "reason": (
            "; ".join(reason)
            if reason
            else "no repeated groups or time structure detected"
        ),
    }


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

            record = {
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
            }

            record = _enrich_record(
                record,
                df,
                series,
                target_column,
            )

            columns.append(record)

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

            elif unique_count <= 10 and not _keep_numeric_role(
                series,
                column,
                unique_count,
            ):
                # Low-cardinality integers (e.g. 0/1, 1-5)
                # behave like categorical labels, unless
                # the column name and values indicate a
                # count or ratio.
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

        record = {
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
        }

        record = _enrich_record(
            record,
            df,
            series,
            target_column,
        )

        columns.append(record)

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

    # ------------------------------------------------
    # AGGREGATED ENRICHED FIELDS
    # ------------------------------------------------

    group_columns = [
        col
        for col in df.columns
        if col != target_column
        and _record_for(columns, col)["group_candidate"]
        and _record_for(columns, col)["group_confidence"]
        in ("high", "medium")
    ]

    semantic_by_type = {}

    for record in columns:

        stype = record.get("semantic_type", record["role"])

        semantic_by_type.setdefault(stype, []).append(
            record["column"]
        )

    high_cardinality_columns = [
        col
        for col in df.columns
        if _record_for(columns, col)["cardinality_level"]
        in ("high", "extreme")
    ]

    target_record = (
        _record_for(columns, target_column)
        if target_column in df.columns
        else None
    )

    validation_strategy = _validation_strategy(
        df,
        group_columns,
        time_column,
        target_column,
        columns,
    )

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
        "target_suitability": (
            target_record.get("target_suitability")
            if target_record
            else None
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
        "group_columns": group_columns,
        "semantic_by_type": semantic_by_type,
        "high_cardinality_columns": high_cardinality_columns,
        "validation_strategy": validation_strategy,
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