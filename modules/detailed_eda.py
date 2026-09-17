import os
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# BASIC HELPERS
# ============================================================

def _detect_id_columns(df):
    """
    Detect identifier-like columns so they are excluded
    from feature analysis.

    A column is treated as an identifier when either:
    - its name clearly suggests an identifier, or
    - it is NOT a continuous numeric feature and almost
      every value is unique (e.g. an object/string key).

    Continuous numeric columns with many unique values are
    NOT treated as identifiers, because features such as
    income, spending or timestamps legitimately have
    near-unique values.
    """

    id_columns = []

    for column in df.columns:

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

        unique_ratio = (
            df[column].nunique(
                dropna=True
            ) / len(df)
        )

        is_numeric = pd.api.types.is_numeric_dtype(
            df[column]
        )

        # Name evidence is decisive
        if name_suggests_id:
            id_columns.append(column)

        # Statistical evidence: near-unique non-numeric key
        elif (
            unique_ratio >= 0.95
            and not is_numeric
        ):
            id_columns.append(column)

    return list(set(id_columns))


def _safe_skew(series):
    """
    Compute skewness without warnings for small data.
    """

    try:
        return round(
            float(series.skew()),
            3
        )

    except Exception:
        return None


def _iqr_outliers(series):
    """
    Detect outlier bounds and counts using the IQR rule.
    """

    valid = series.dropna()

    if len(valid) == 0:
        return 0, None, None

    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))

    iqr = q3 - q1

    if iqr == 0:
        return 0, q1, q3

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = int(
        ((valid < lower) | (valid > upper)).sum()
    )

    return count, round(lower, 3), round(upper, 3)


# ============================================================
# TARGET ANALYSIS
# ============================================================

def _analyze_classification_target(
    df,
    target_column
):
    """
    Class distribution, imbalance and rare classes.
    """

    series = df[target_column].dropna()

    total = len(series)

    counts = series.value_counts()

    distribution = []

    for label, count in counts.items():

        distribution.append({
            "class": str(label),
            "count": int(count),
            "percentage": round(
                count / total * 100,
                2
            )
        })

    imbalance = None

    if len(counts) >= 2:

        majority = counts.index[0]
        minority = counts.index[-1]

        majority_count = int(counts.iloc[0])
        minority_count = int(counts.iloc[-1])

        ratio = (
            majority_count / minority_count
            if minority_count > 0
            else None
        )

        imbalance = {
            "majority_class": str(majority),
            "majority_count": majority_count,
            "majority_percentage": round(
                majority_count / total * 100,
                2
            ),
            "minority_class": str(minority),
            "minority_count": minority_count,
            "minority_percentage": round(
                minority_count / total * 100,
                2
            ),
            "imbalance_ratio": (
                round(ratio, 2)
                if ratio is not None
                else None
            )
        }

    # Rare classes: very small percentage of the total
    rare_classes = [
        item
        for item in distribution
        if item["percentage"] < 5
    ]

    return {
        "total": total,
        "class_count": int(series.nunique()),
        "distribution": distribution,
        "imbalance": imbalance,
        "rare_classes": rare_classes
    }


def _analyze_regression_target(
    df,
    target_column
):
    """
    Target statistics, distribution shape and outliers.
    """

    series = df[target_column].dropna()

    stats = series.describe()

    outliers_count, lower, upper = _iqr_outliers(
        series
    )

    return {
        "total": len(series),
        "mean": round(float(stats.get("mean", 0)), 3),
        "median": round(float(series.median()), 3),
        "std": round(float(stats.get("std", 0)), 3),
        "min": round(float(stats.get("min", 0)), 3),
        "max": round(float(stats.get("max", 0)), 3),
        "q1": round(float(series.quantile(0.25)), 3),
        "q3": round(float(series.quantile(0.75)), 3),
        "range": round(
            float(stats.get("max", 0))
            - float(stats.get("min", 0)),
            3
        ),
        "variance": round(
            float(series.var()),
            3
        ),
        "skewness": _safe_skew(series),
        "outliers_count": outliers_count,
        "outlier_bounds": {
            "lower": lower,
            "upper": upper
        }
    }


# ============================================================
# FEATURE ANALYSIS
# ============================================================

def _analyze_numerical_features(
    df,
    feature_columns,
    target_column=None,
    problem_type="Classification"
):
    """
    Statistics for every numerical feature, including
    feature-target relationship.
    """

    analysis = {}

    for column in feature_columns:

        series = df[column].dropna()

        stats = series.describe()

        outliers_count, lower, upper = _iqr_outliers(
            series
        )

        item = {
            "column": column,
            "count": int(stats.get("count", 0)),
            "mean": round(float(stats.get("mean", 0)), 3),
            "median": round(float(series.median()), 3),
            "std": round(float(stats.get("std", 0)), 3),
            "min": round(float(stats.get("min", 0)), 3),
            "max": round(float(stats.get("max", 0)), 3),
            "q1": round(float(series.quantile(0.25)), 3),
            "q3": round(float(series.quantile(0.75)), 3),
            "skewness": _safe_skew(series),
            "missing": int(
                df[column].isnull().sum()
            ),
            "outliers_count": outliers_count,
            "outlier_bounds": {
                "lower": lower,
                "upper": upper
            }
        }

        # ------------------------------------------------
        # FEATURE TARGET RELATIONSHIP
        # ------------------------------------------------

        if target_column is not None:

            target = df[target_column].dropna()

            if problem_type == "Classification":

                grouped = (
                    df[[column, target_column]]
                    .dropna()
                    .groupby(target_column)[column]
                    .mean()
                )

                item["group_means"] = [
                    {
                        "class": str(label),
                        "mean": round(float(value), 3)
                    }
                    for label, value
                    in grouped.items()
                ]

            else:

                target = df[[target_column, column]].dropna()

                if len(target) >= 3:

                    corr = target[column].corr(
                        target[target_column]
                    )

                    item["correlation_with_target"] = (
                        round(float(corr), 3)
                        if not math.isnan(corr)
                        else None
                    )

        analysis[column] = item

    return analysis


def _analyze_categorical_features(
    df,
    feature_columns,
    target_column=None
):
    """
    Frequency analysis for categorical features.
    """

    analysis = {}

    for column in feature_columns:

        series = df[column].dropna()

        total = len(series)

        counts = series.value_counts()

        top_categories = []

        for label, count in counts.head(8).items():

            top_categories.append({
                "category": str(label),
                "count": int(count),
                "percentage": round(
                    count / total * 100,
                    2
                )
            })

        rare_categories = [
            str(label)
            for label, count in counts.items()
            if count / total < 0.05
        ]

        item = {
            "column": column,
            "unique_count": int(series.nunique()),
            "total": total,
            "missing": int(df[column].isnull().sum()),
            "top_categories": top_categories,
            "rare_categories": rare_categories[:10]
        }

        analysis[column] = item

    return analysis


# ============================================================
# CORRELATION
# ============================================================

def _analyze_correlation(df, feature_columns):
    """
    Correlation matrix among numerical features.
    """

    numerical = df[feature_columns].select_dtypes(
        include=[np.number]
    )

    if numerical.shape[1] < 2:
        return []

    matrix = numerical.corr()

    rows = []

    for i, col_a in enumerate(matrix.columns):

        for col_b in matrix.columns[i + 1:]:

            value = matrix.loc[col_a, col_b]

            if math.isnan(value):
                continue

            rows.append({
                "feature_a": str(col_a),
                "feature_b": str(col_b),
                "correlation": round(float(value), 3)
            })

    rows.sort(
        key=lambda item: abs(item["correlation"]),
        reverse=True
    )

    return rows[:15]


# ============================================================
# CHART GENERATION
# ============================================================

def _generate_target_chart(
    df,
    target_column,
    problem_type,
    output_path
):
    """
    Generate a target distribution chart.
    """

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    series = df[target_column].dropna()

    if problem_type == "Classification":

        counts = series.value_counts()

        ax.bar(
            [str(label) for label in counts.index],
            counts.values,
            color="#4C72B0"
        )

        ax.set_title(
            f"Target Distribution: {target_column}"
        )

        ax.set_ylabel("Count")

        ax.set_xlabel("Class")

        for i, value in enumerate(counts.values):

            ax.text(
                i,
                value,
                str(int(value)),
                ha="center",
                va="bottom"
            )

        ax.tick_params(
            axis="x",
            rotation=45
        )

    else:

        ax.hist(
            series,
            bins=min(30, int(series.nunique())),
            color="#4C72B0",
            edgecolor="white"
        )

        ax.set_title(
            f"Target Distribution: {target_column}"
        )

        ax.set_ylabel("Frequency")

        ax.set_xlabel("Value")

    fig.tight_layout()

    fig.savefig(output_path, dpi=100)

    plt.close(fig)


def _generate_correlation_chart(
    matrix,
    output_path
):
    """
    Generate a correlation heatmap.
    """

    if matrix.shape[0] < 2:
        return False

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        matrix.values,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(
        range(len(matrix.columns))
    )

    ax.set_yticks(
        range(len(matrix.index))
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
        fontsize=8
    )

    ax.set_yticklabels(
        matrix.index,
        fontsize=8
    )

    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            value = matrix.iloc[i, j]

            if math.isnan(value):
                continue

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=(
                    "black"
                    if abs(value) < 0.6
                    else "white"
                )
            )

    ax.set_title("Feature Correlation Matrix")

    fig.colorbar(im)

    fig.tight_layout()

    fig.savefig(output_path, dpi=100)

    plt.close(fig)

    return True


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def perform_detailed_eda(
    df,
    target_column=None,
    problem_type="Classification",
    charts_dir=None,
    chart_id=None
):
    """
    Perform target/task-specific detailed EDA.

    Parameters
    ----------
    df : pandas.DataFrame
        The uploaded dataset.
    target_column : str or None
        The confirmed target. None for clustering mode.
    problem_type : str
        Classification, Regression or Clustering.
    charts_dir : str or None
        Directory where chart images are saved.
    chart_id : str or None
        Prefix for chart file names.

    Returns
    -------
    dict
        Structured EDA results ready for template rendering.
    """

    # Normalize empty target to None (clustering mode)
    if target_column in (None, ""):
        target_column = None

    if problem_type == "Clustering":
        target_column = None

    id_columns = _detect_id_columns(df)

    # Constant columns carry no predictive information
    constant_columns = [
        column
        for column in df.columns
        if column not in id_columns
        and column != target_column
        and df[column].nunique(
            dropna=True
        ) <= 1
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in id_columns
        and column not in constant_columns
        and column != target_column
    ]

    numerical_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    categorical_columns = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    # ------------------------------------------------
    # TARGET ANALYSIS
    # ------------------------------------------------

    target_analysis = None

    if target_column is not None:

        if problem_type == "Classification":

            target_analysis = (
                _analyze_classification_target(
                    df,
                    target_column
                )
            )

        else:

            target_analysis = (
                _analyze_regression_target(
                    df,
                    target_column
                )
            )

    # ------------------------------------------------
    # FEATURE ANALYSIS
    # ------------------------------------------------

    numerical_analysis = _analyze_numerical_features(
        df,
        numerical_columns,
        target_column=target_column,
        problem_type=problem_type
    )

    categorical_analysis = _analyze_categorical_features(
        df,
        categorical_columns,
        target_column=target_column
    )

    correlation = _analyze_correlation(
        df,
        feature_columns
    )

    # ------------------------------------------------
    # CHARTS
    # ------------------------------------------------

    charts = []

    if charts_dir:

        os.makedirs(
            charts_dir,
            exist_ok=True
        )

        base = (
            chart_id
            if chart_id
            else "eda"
        )

        if target_column is not None:

            target_chart = os.path.join(
                charts_dir,
                f"{base}_target.png"
            )

            _generate_target_chart(
                df,
                target_column,
                problem_type,
                target_chart
            )

            charts.append({
                "name": "Target Distribution",
                "file": f"{base}_target.png",
                "caption": (
                    "Distribution of the confirmed target "
                    "variable."
                )
            })

        numerical_subset = df[numerical_columns].select_dtypes(
            include=[np.number]
        )

        if numerical_subset.shape[1] >= 2:

            corr_matrix = numerical_subset.corr()

            corr_chart = os.path.join(
                charts_dir,
                f"{base}_correlation.png"
            )

            if _generate_correlation_chart(
                corr_matrix,
                corr_chart
            ):

                charts.append({
                    "name": "Feature Correlation",
                    "file": f"{base}_correlation.png",
                    "caption": (
                        "Correlation among numerical "
                        "features. Strong values may "
                        "indicate redundancy."
                    )
                })

    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------

    return {
        "target_column": target_column,
        "problem_type": problem_type,
        "id_columns": id_columns,
        "constant_columns": constant_columns,
        "feature_columns": feature_columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "target_analysis": target_analysis,
        "numerical_analysis": numerical_analysis,
        "categorical_analysis": categorical_analysis,
        "correlation": correlation,
        "charts": charts
    }