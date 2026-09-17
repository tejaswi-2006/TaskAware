import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold


# ============================================================
# HIGH CORRELATION FEATURES
# ============================================================

def find_high_correlation(
    df,
    feature_columns,
    threshold=0.90
):
    """
    Identify pairs of numerical features with very high
    correlation (potential redundancy).
    """

    numerical = df[feature_columns].select_dtypes(
        include=[np.number]
    )

    pairs = []

    if numerical.shape[1] < 2:
        return pairs

    matrix = numerical.corr()

    seen = set()

    for i, col_a in enumerate(matrix.columns):

        for col_b in matrix.columns[i + 1:]:

            value = matrix.loc[col_a, col_b]

            if np.isnan(value):
                continue

            if abs(value) >= threshold:

                pair = tuple(
                    sorted([str(col_a), str(col_b)])
                )

                if pair in seen:
                    continue

                seen.add(pair)

                pairs.append({
                    "feature_a": str(col_a),
                    "feature_b": str(col_b),
                    "correlation": round(
                        float(value),
                        3
                    ),
                    "redundancy_level": (
                        "Very High"
                        if abs(value) >= 0.95
                        else "High"
                    )
                })

    pairs.sort(
        key=lambda item: abs(item["correlation"]),
        reverse=True
    )

    return pairs


# ============================================================
# LOW VARIANCE / NEAR CONSTANT
# ============================================================

def find_low_variance(
    df,
    numerical_columns,
    categorical_columns,
    threshold=0.01
):
    """
    Identify features with very low variance after
    standardization. These carry little information.
    """

    low = []

    for column in numerical_columns:

        series = df[column].dropna()

        if len(series) < 2:
            continue

        std = float(series.std())

        if std == 0:
            low.append({
                "column": column,
                "std": 0.0,
                "note": "Zero variance (constant)"
            })
        elif std < threshold:
            low.append({
                "column": column,
                "std": round(std, 4),
                "note": "Very low variance"
            })

    # Categorical: near-constant (one dominant category)
    for column in categorical_columns:

        series = df[column].dropna()

        if len(series) == 0:
            continue

        dominant_ratio = (
            series.value_counts(
                normalize=True
            ).iloc[0]
        )

        if dominant_ratio >= 0.99:

            low.append({
                "column": column,
                "std": None,
                "note": (
                    f"Near-constant: one category covers "
                    f"{dominant_ratio:.1%} of values"
                )
            })

    return low


# ============================================================
# LEAKAGE CANDIDATES
# ============================================================

def find_leakage_candidates(
    df,
    feature_columns,
    target_column,
    threshold=0.97
):
    """
    Identify features suspiciously correlated with the
    target. These may leak the outcome.

    Only checks numerical-vs-numerical correlation and
    is reported as a concern, not an automatic removal.
    """

    if not target_column:
        return []

    target = df[target_column]

    if not pd.api.types.is_numeric_dtype(target):
        return []

    candidates = []

    for column in feature_columns:

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        if column == target_column:
            continue

        valid = df[[column, target_column]].dropna()

        if len(valid) < 3:
            continue

        corr = valid[column].corr(valid[target_column])

        if np.isnan(corr):
            continue

        if abs(corr) >= threshold:

            candidates.append({
                "column": column,
                "correlation_with_target": round(
                    float(corr),
                    3
                ),
                "warning": (
                    f"Correlation of {corr:.2f} with the "
                    "target may indicate information leakage "
                    "if this feature is only known after the "
                    "outcome occurs."
                )
            })

    candidates.sort(
        key=lambda item: abs(
            item["correlation_with_target"]
        ),
        reverse=True
    )

    return candidates


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def perform_feature_selection(
    df,
    feature_columns,
    target_column=None,
    high_corr_threshold=0.90
):
    """
    Analyze features and report selection-relevant findings.

    TaskAware does NOT automatically delete features based
    on low correlation. It only reports findings and removes
    features with no information at all (already handled in
    preprocessing: ids and constant columns).

    Returns
    -------
    dict
        Structured feature-selection report.
    """

    high_correlation = find_high_correlation(
        df,
        feature_columns,
        threshold=high_corr_threshold
    )

    numerical = [
        c for c in feature_columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    categorical = [
        c for c in feature_columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    low_variance = find_low_variance(
        df,
        numerical,
        categorical
    )

    leakage = find_leakage_candidates(
        df,
        feature_columns,
        target_column
    )

    # Features with no apparent linear relationship
    # are NOT removed. This is reported as guidance only.
    return {
        "feature_count": len(feature_columns),
        "high_correlation": high_correlation,
        "high_corr_count": len(high_correlation),
        "low_variance": low_variance,
        "low_variance_count": len(low_variance),
        "leakage_candidates": leakage,
        "leakage_count": len(leakage)
    }