import pandas as pd


def detect_problem_type(
    df,
    target_column,
    objective=None
):
    """
    Detect whether the selected target represents
    classification or regression.

    Decision is based on:
    - target data type
    - number of unique values
    - target distribution characteristics
    - optional objective information
    """

    # ========================================================
    # CHECK TARGET
    # ========================================================

    if target_column not in df.columns:

        return {

            "success": False,

            "problem_type": None,

            "message":
                "Selected target column does not exist."
        }

    target = df[
        target_column
    ].dropna()

    # ========================================================
    # EMPTY TARGET
    # ========================================================

    if target.empty:

        return {

            "success": False,

            "problem_type": None,

            "target_column":
                target_column,

            "message":
                "Target column contains no valid values."
        }

    # ========================================================
    # TARGET INFORMATION
    # ========================================================

    unique_count = int(
        target.nunique()
    )

    total_count = len(target)

    unique_ratio = (
        unique_count /
        total_count
    )

    # ========================================================
    # DATA TYPE
    # ========================================================

    if pd.api.types.is_numeric_dtype(target):

        data_type = "numerical"

    else:

        data_type = "categorical"

    # ========================================================
    # OBJECTIVE HINT
    # ========================================================

    objective_text = ""

    if objective:

        objective_text = (
            str(objective)
            .lower()
            .strip()
        )

    # ========================================================
    # CATEGORICAL TARGET
    # ========================================================

    if data_type == "categorical":

        return {

            "success": True,

            "problem_type":
                "Classification",

            "target_column":
                target_column,

            "data_type":
                data_type,

            "unique_values":
                unique_count,

            "message": (
                "The selected target is categorical, "
                "so TaskAware treats the problem as "
                "classification."
            )
        }

    # ========================================================
    # NUMERICAL TARGET
    # ========================================================

    if data_type == "numerical":

        # ----------------------------------------------------
        # Explicit regression objective
        # ----------------------------------------------------

        regression_words = [
            "regression",
            "predict score",
            "predict price",
            "predict salary",
            "predict income",
            "predict sales",
            "predict amount",
            "predict value"
        ]

        classification_words = [
            "classify",
            "classification",
            "categorize",
            "categorise",
            "category",
            "class",
            "label"
        ]

        objective_requests_regression = any(
            word in objective_text
            for word in regression_words
        )

        objective_requests_classification = any(
            word in objective_text
            for word in classification_words
        )

        # ----------------------------------------------------
        # Numerical target with few classes
        # ----------------------------------------------------

        if (
            unique_count <= 10
            and unique_ratio < 0.05
        ):

            # If objective explicitly asks for regression,
            # respect that and treat the target as numerical.
            if objective_requests_regression:

                return {

                    "success": True,

                    "problem_type":
                        "Regression",

                    "target_column":
                        target_column,

                    "data_type":
                        data_type,

                    "unique_values":
                        unique_count,

                    "message": (
                        "The target is numerical with a "
                        "small number of distinct values. "
                        "The stated objective indicates "
                        "that numerical prediction is intended, "
                        "so TaskAware treats it as regression."
                    )
                }

            return {

                "success": True,

                "problem_type":
                    "Classification",

                "target_column":
                    target_column,

                "data_type":
                    data_type,

                "unique_values":
                    unique_count,

                "message": (
                    "The numerical target contains a small "
                    "number of distinct values and may represent "
                    "class labels. TaskAware treats it as "
                    "classification unless the objective indicates "
                    "a numerical prediction."
                )
            }

        # ----------------------------------------------------
        # Continuous numerical target
        # ----------------------------------------------------

        return {

            "success": True,

            "problem_type":
                "Regression",

            "target_column":
                target_column,

            "data_type":
                data_type,

            "unique_values":
                unique_count,

            "message": (
                "The selected target contains numerical "
                "values with sufficient variation, so "
                "TaskAware treats the problem as regression."
            )
        }

    # ========================================================
    # UNKNOWN
    # ========================================================

    return {

        "success": False,

        "problem_type": None,

        "target_column":
            target_column,

        "data_type":
            data_type,

        "unique_values":
            unique_count,

        "message":
            "Unable to determine the problem type."
    }