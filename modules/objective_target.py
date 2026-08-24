import pandas as pd


def analyze_target_candidates(df):
    """
    Analyze columns and rank their suitability
    as possible target variables.
    """

    candidates = []

    total_rows = len(df)

    # Words that often indicate an output/outcome/target
    target_keywords = [
        "target",
        "label",
        "class",
        "result",
        "outcome",
        "final",
        "grade",
        "score",
        "performance",
        "status",
        "prediction",
        "disease",
        "churn",
        "price",
        "sales",
        "income"
    ]

    # Words that often indicate input features
    feature_keywords = [
        "id",
        "gender",
        "sex",
        "age",
        "access",
        "activity",
        "education",
        "job",
        "hours",
        "attendance",
        "sleep",
        "previous"
    ]

    for column in df.columns:

        unique_count = df[column].nunique()
        missing_count = df[column].isnull().sum()

        unique_ratio = (
            unique_count / total_rows
            if total_rows > 0
            else 0
        )

        column_name = column.lower()

        dtype = str(df[column].dtype)

        # -----------------------------
        # DETECT COLUMN TYPE
        # -----------------------------

        if dtype in ["object", "category", "bool"]:

            column_type = "categorical"

        elif pd.api.types.is_numeric_dtype(df[column]):

            if unique_count <= 10:
                column_type = "categorical_numeric"
            else:
                column_type = "numerical"

        else:
            column_type = "other"

        # -----------------------------
        # TARGET SCORE
        # -----------------------------

        score = 0
        reasons = []

        # Categorical target candidate
        if column_type in [
            "categorical",
            "categorical_numeric"
        ]:

            if 2 <= unique_count <= 10:

                score += 30

                reasons.append(
                    "Has a suitable number of categories"
                )

        # Numerical target candidate
        elif column_type == "numerical":

            score += 20

            reasons.append(
                "Contains multiple numerical values"
            )

        # -----------------------------
        # CHECK TARGET KEYWORDS
        # -----------------------------

        found_target_keyword = False

        for keyword in target_keywords:

            if keyword in column_name:

                score += 50

                found_target_keyword = True

                reasons.append(
                    f"Column name contains target-related keyword: '{keyword}'"
                )

                break

        # -----------------------------
        # CHECK FEATURE KEYWORDS
        # -----------------------------

        for keyword in feature_keywords:

            if keyword in column_name:

                score -= 30

                reasons.append(
                    f"Column appears to describe an input feature: '{keyword}'"
                )

                break

        # -----------------------------
        # ID COLUMN PENALTY
        # -----------------------------

        if unique_ratio > 0.95:

            score -= 50

            reasons.append(
                "Most values are unique; likely an ID column"
            )

        # -----------------------------
        # MISSING VALUE PENALTY
        # -----------------------------

        missing_ratio = (
            missing_count / total_rows
            if total_rows > 0
            else 0
        )

        if missing_ratio > 0.30:

            score -= 20

            reasons.append(
                "Contains a high percentage of missing values"
            )

        # -----------------------------
        # BINARY CLASSIFICATION BONUS
        # -----------------------------

        if (
            column_type == "categorical"
            and unique_count == 2
        ):

            score += 10

            reasons.append(
                "Suitable for binary classification"
            )

        candidates.append({

            "column": column,

            "type": column_type,

            "unique_values": int(unique_count),

            "missing_values": int(missing_count),

            "score": score,

            "reasons": reasons
        })

    # Sort highest score first
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates