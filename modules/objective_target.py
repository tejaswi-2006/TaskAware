import pandas as pd
import re


# ============================================================
# OBJECTIVE ANALYSIS
# ============================================================

def analyze_objective(objective):
    """
    Analyze the user's objective.

    The objective is used as a supporting signal.
    Dataset characteristics remain the primary evidence.
    """

    if not objective:

        return {
            "task_hint": None,
            "concepts": [],
            "text": ""
        }

    text = objective.lower().strip()

    # --------------------------------------------------------
    # TASK HINTS
    # --------------------------------------------------------

    classification_patterns = [

        r"\bclassify\b",
        r"\bclassification\b",
        r"\bcategorize\b",
        r"\bcategorise\b",
        r"\bcategory\b",
        r"\bgrade\b",
        r"\bstatus\b",
        r"\bdisease\b",
        r"\bdiagnos",
        r"\bpass\b",
        r"\bfail\b",
        r"\bchurn\b",
        r"\blabel\b"
    ]

    regression_patterns = [

        r"\bregression\b",
        r"\bpredict.*score\b",
        r"\bpredict.*price\b",
        r"\bpredict.*salary\b",
        r"\bpredict.*income\b",
        r"\bpredict.*sales\b",
        r"\bpredict.*amount\b",
        r"\bpredict.*value\b",
        r"\bpredict.*temperature\b"
    ]

    classification_match = any(
        re.search(
            pattern,
            text
        )
        for pattern in classification_patterns
    )

    regression_match = any(
        re.search(
            pattern,
            text
        )
        for pattern in regression_patterns
    )

    if classification_match and not regression_match:

        task_hint = "Classification"

    elif regression_match and not classification_match:

        task_hint = "Regression"

    else:

        task_hint = None

    # --------------------------------------------------------
    # OBJECTIVE CONCEPTS
    # --------------------------------------------------------

    concept_groups = {

        "performance": [
            "performance",
            "achievement",
            "academic performance",
            "student performance"
        ],

        "grade": [
            "grade",
            "grades"
        ],

        "score": [
            "score",
            "scores",
            "marks",
            "mark",
            "exam score",
            "test score"
        ],

        "result": [
            "result",
            "results",
            "outcome",
            "outcomes"
        ],

        "price": [
            "price",
            "cost",
            "amount"
        ],

        "sales": [
            "sales",
            "revenue"
        ],

        "income": [
            "income",
            "salary"
        ],

        "status": [
            "status"
        ],

        "disease": [
            "disease",
            "diagnosis",
            "diagnose"
        ],

        "churn": [
            "churn"
        ]
    }

    concepts = []

    for concept, words in concept_groups.items():

        if any(
            word in text
            for word in words
        ):

            concepts.append(
                concept
            )

    return {

        "task_hint": task_hint,

        "concepts": list(
            set(concepts)
        ),

        "text": text
    }


# ============================================================
# COLUMN NAME NORMALIZATION
# ============================================================

def normalize_column_name(column):

    text = str(
        column
    ).strip().lower()

    text = text.replace(
        "_",
        " "
    )

    text = text.replace(
        "-",
        " "
    )

    return re.findall(
        r"[a-zA-Z0-9]+",
        text
    )


# ============================================================
# IDENTIFIER DETECTION
# ============================================================

def detect_identifier(
    series,
    column_name,
    total_rows
):
    """
    Detect identifier-like columns using both
    statistical evidence and column naming.
    """

    if total_rows == 0:

        return False

    unique_count = int(
        series.nunique(
            dropna=True
        )
    )

    unique_ratio = (
        unique_count /
        total_rows
    )

    tokens = normalize_column_name(
        column_name
    )

    id_words = {

        "id",
        "identifier",
        "studentid",
        "customerid",
        "userid",
        "recordid",
        "userid",
        "rollno",
        "rollnumber",
        "registrationid",
        "serial",
        "serialnumber"
    }

    name_suggests_id = any(

        token in id_words

        for token in tokens
    )

    # Strong statistical evidence
    if unique_ratio >= 0.95:

        return True

    # Strong naming evidence
    if name_suggests_id:

        return True

    return False


# ============================================================
# COLUMN TYPE
# ============================================================

def determine_column_type(series):

    if pd.api.types.is_bool_dtype(series):

        return "categorical"

    if pd.api.types.is_numeric_dtype(series):

        unique_count = series.nunique(
            dropna=True
        )

        if unique_count <= 10:

            return "categorical_numeric"

        return "numerical"

    if (
        pd.api.types.is_object_dtype(series)
        or
        isinstance(
            series.dtype,
            pd.CategoricalDtype
        )
    ):

        return "categorical"

    return "other"


# ============================================================
# OBJECTIVE / COLUMN ALIGNMENT
# ============================================================

def objective_alignment(
    column,
    objective_info
):
    """
    Provide a small objective-related score.

    This must NOT dominate statistical evidence.
    """

    if not objective_info["concepts"]:

        return 0, []

    column_text = (
        str(column)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    score = 0
    reasons = []

    concepts = objective_info[
        "concepts"
    ]

    # --------------------------------------------------------
    # PERFORMANCE OBJECTIVE
    # --------------------------------------------------------

    if "performance" in concepts:

        if any(
            word in column_text
            for word in [
                "performance",
                "achievement"
            ]
        ):

            score += 15

            reasons.append(
                "The column directly represents the "
                "performance concept in the objective."
            )

        elif "final" in column_text and any(
            word in column_text
            for word in [
                "grade",
                "score",
                "result",
                "outcome"
            ]
        ):

            score += 15

            reasons.append(
                "The column appears to represent a final "
                "academic outcome related to the objective."
            )

        elif any(
            word in column_text
            for word in [
                "grade",
                "score",
                "result",
                "outcome"
            ]
        ):

            score += 10

            reasons.append(
                "The column represents an outcome related "
                "to the performance objective."
            )

    # --------------------------------------------------------
    # GRADE
    # --------------------------------------------------------

    if "grade" in concepts:

        if "grade" in column_text:

            score += 15

            reasons.append(
                "The column directly matches the grade "
                "concept in the objective."
            )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    if "score" in concepts:

        if any(
            word in column_text
            for word in [
                "score",
                "marks",
                "mark"
            ]
        ):

            score += 15

            reasons.append(
                "The column directly matches the score "
                "concept in the objective."
            )

    # --------------------------------------------------------
    # RESULT / OUTCOME
    # --------------------------------------------------------

    if "result" in concepts:

        if any(
            word in column_text
            for word in [
                "result",
                "outcome"
            ]
        ):

            score += 15

            reasons.append(
                "The column represents an outcome related "
                "to the stated objective."
            )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    if "price" in concepts:

        if any(
            word in column_text
            for word in [
                "price",
                "cost",
                "amount"
            ]
        ):

            score += 15

            reasons.append(
                "The column matches the price/value "
                "concept in the objective."
            )

    # --------------------------------------------------------
    # SALES
    # --------------------------------------------------------

    if "sales" in concepts:

        if any(
            word in column_text
            for word in [
                "sales",
                "revenue"
            ]
        ):

            score += 15

            reasons.append(
                "The column matches the sales/revenue "
                "concept in the objective."
            )

    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    if "income" in concepts:

        if any(
            word in column_text
            for word in [
                "income",
                "salary"
            ]
        ):

            score += 15

            reasons.append(
                "The column matches the income/salary "
                "concept in the objective."
            )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if "status" in concepts:

        if "status" in column_text:

            score += 15

            reasons.append(
                "The column directly represents status "
                "information described by the objective."
            )

    # --------------------------------------------------------
    # DISEASE
    # --------------------------------------------------------

    if "disease" in concepts:

        if any(
            word in column_text
            for word in [
                "disease",
                "diagnosis"
            ]
        ):

            score += 15

            reasons.append(
                "The column represents the disease/diagnosis "
                "concept in the objective."
            )

    # --------------------------------------------------------
    # CHURN
    # --------------------------------------------------------

    if "churn" in concepts:

        if "churn" in column_text:

            score += 15

            reasons.append(
                "The column directly represents customer "
                "churn, which is the concept described in "
                "the objective."
            )

    return min(
        score,
        25
    ), reasons


# ============================================================
# MAIN TARGET ANALYSIS
# ============================================================

def analyze_target_candidates(
    df,
    objective=None
):
    """
    Analyze and rank possible target columns.

    Primary evidence:
    - data characteristics
    - cardinality
    - missingness
    - identifier likelihood
    - variation

    Supporting evidence:
    - objective alignment
    - column naming
    """

    candidates = []

    total_rows = len(df)

    if total_rows == 0:

        return candidates

    objective_info = analyze_objective(
        objective
    )

    for column in df.columns:

        series = df[column]

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        missing_count = int(
            series.isnull().sum()
        )

        missing_ratio = (
            missing_count /
            total_rows
        )

        unique_ratio = (
            unique_count /
            total_rows
        )

        column_type = determine_column_type(
            series
        )

        score = 0

        positive_reasons = []

        negative_reasons = []

        # ====================================================
        # 1. IDENTIFIER
        # ====================================================

        is_id = detect_identifier(

            series,

            column,

            total_rows
        )

        if is_id:

            score -= 100

            if unique_ratio >= 0.95:

                negative_reasons.append(

                    f"Almost every value is unique "
                    f"({unique_count} out of "
                    f"{total_rows} rows), indicating "
                    "identifier-like behavior."
                )

            else:

                negative_reasons.append(
                    "The column name suggests that "
                    "this may be an identifier."
                )

        # ====================================================
        # 2. EMPTY / CONSTANT
        # ====================================================

        if unique_count == 0:

            score -= 100

            negative_reasons.append(
                "The column contains no valid values."
            )

        elif unique_count == 1:

            score -= 100

            negative_reasons.append(
                "The column contains only one unique "
                "value and cannot provide meaningful "
                "target variation."
            )

        # ====================================================
        # 3. MISSING VALUES
        # ====================================================

        if missing_ratio == 0:

            score += 10

            positive_reasons.append(
                "The target contains no missing values."
            )

        elif missing_ratio <= 0.10:

            score += 5

            positive_reasons.append(

                f"The target has a relatively small "
                f"amount of missing data "
                f"({missing_ratio:.1%})."
            )

        elif missing_ratio <= 0.30:

            negative_reasons.append(

                f"The target contains "
                f"{missing_ratio:.1%} missing values "
                "and would require handling."
            )

        else:

            score -= 30

            negative_reasons.append(

                f"The target has a high proportion "
                f"of missing values "
                f"({missing_ratio:.1%})."
            )

        # ====================================================
        # 4. CATEGORICAL TARGET
        # ====================================================

        if column_type in [
            "categorical",
            "categorical_numeric"
        ]:

            if 2 <= unique_count <= 10:

                score += 30

                positive_reasons.append(

                    f"The target has {unique_count} "
                    "distinct classes, which is "
                    "appropriate for classification."
                )

            elif unique_count > 10:

                score -= 15

                negative_reasons.append(

                    f"The target has {unique_count} "
                    "distinct categories, which may "
                    "make classification more difficult."
                )

        # ====================================================
        # 5. NUMERICAL TARGET
        # ====================================================

        elif column_type == "numerical":

            if unique_count >= 20:

                score += 30

                positive_reasons.append(

                    f"The numerical target has "
                    f"{unique_count} distinct values, "
                    "providing useful variation for "
                    "regression."
                )

            elif unique_count > 10:

                score += 15

                positive_reasons.append(
                    "The numerical target has moderate "
                    "variation and may be suitable "
                    "for regression."
                )

            else:

                score -= 5

                negative_reasons.append(
                    "The numerical target has limited "
                    "variation."
                )

        # ====================================================
        # 6. OBJECTIVE TASK ALIGNMENT
        # ====================================================

        task_hint = objective_info[
            "task_hint"
        ]

        if task_hint == "Classification":

            if column_type in [
                "categorical",
                "categorical_numeric"
            ] and 2 <= unique_count <= 10:

                score += 25

                positive_reasons.append(

                    "The target type is consistent with "
                    "the classification requirement "
                    "in the objective."
                )

            elif column_type == "numerical":

                score -= 5

                negative_reasons.append(

                    "The target is numerical while "
                    "the objective appears to describe "
                    "a classification task."
                )

        elif task_hint == "Regression":

            if column_type == "numerical":

                score += 25

                positive_reasons.append(

                    "The numerical target is consistent "
                    "with the regression requirement "
                    "in the objective."
                )

            elif column_type in [
                "categorical",
                "categorical_numeric"
            ]:

                score -= 5

                negative_reasons.append(

                    "The target is categorical while "
                    "the objective appears to require "
                    "numerical prediction."
                )

        # ====================================================
        # 7. OBJECTIVE / COLUMN ALIGNMENT
        # ====================================================

        alignment_score, alignment_reasons = (
            objective_alignment(

                column,

                objective_info
            )
        )

        score += alignment_score

        positive_reasons.extend(
            alignment_reasons
        )

        # ====================================================
        # 8. GENERIC TARGET NAME
        # ====================================================

        target_words = {

            "target",
            "label",
            "result",
            "outcome",
            "grade",
            "score",
            "status",
            "price",
            "sales",
            "income",
            "performance",
            "churn",
            "disease"
        }

        column_text = (
            str(column)
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        matched_target_words = [
            word
            for word in target_words
            if word in column_text
        ]

        if matched_target_words:

            score += 5

            positive_reasons.append(

                "The column name suggests that it "
                "may represent an outcome or target."
            )

        # ====================================================
        # 9. FEATURE-LIKE NAME
        # ====================================================

        feature_words = {

            "gender",
            "sex",
            "age",
            "study",
            "attendance",
            "sleep",
            "education",
            "job",
            "hours",
            "internet",
            "activity"
        }

        if any(
            word in column_text
            for word in feature_words
        ):

            score -= 3

            negative_reasons.append(

                "The column name suggests that it "
                "may represent an input feature rather "
                "than an outcome."
            )

        # ====================================================
        # 10. SUITABILITY
        # ====================================================

        if is_id:

            suitability = "Not Recommended"

        elif unique_count <= 1:

            suitability = "Not Recommended"

        elif score >= 70:

            suitability = "Highly Suitable"

        elif score >= 45:

            suitability = "Suitable"

        elif score >= 20:

            suitability = "Possible"

        else:

            suitability = "Low Suitability"

        # ====================================================
        # 11. POSSIBLE PROBLEM TYPE
        # ====================================================

        if column_type in [
            "categorical",
            "categorical_numeric"
        ]:

            possible_problem_type = (
                "Classification"
            )

        elif column_type == "numerical":

            possible_problem_type = (
                "Regression"
            )

        else:

            possible_problem_type = (
                "Unknown"
            )

        # ====================================================
        # 12. CONFIDENCE
        # ====================================================

        if suitability == "Highly Suitable":

            confidence = "High"

        elif suitability == "Suitable":

            confidence = "Moderate"

        elif suitability == "Possible":

            confidence = "Low"

        else:

            confidence = "Very Low"

        # ====================================================
        # 13. FINAL RESULT
        # ====================================================

        candidates.append({

            "column": column,

            "type": column_type,

            "unique_values": unique_count,

            "unique_ratio": round(
                unique_ratio,
                3
            ),

            "missing_values": missing_count,

            "missing_ratio": round(
                missing_ratio,
                3
            ),

            "score": int(score),

            "suitability": suitability,

            "confidence": confidence,

            "possible_problem_type":
                possible_problem_type,

            "positive_reasons":
                positive_reasons,

            "negative_reasons":
                negative_reasons,

            "reasons":
                positive_reasons +
                negative_reasons,

            "is_id": is_id
        })

    # ========================================================
    # SORT
    # ========================================================

    candidates.sort(
        key=lambda item: (
            item["score"],
            # Tie-break: fewer missing values is preferred
            -item["missing_ratio"],
            # Tie-break: a lower unique ratio is preferred
            # because it implies more balanced classes
            # and is less likely to be identifier-like.
            -item["unique_ratio"]
        ),
        reverse=True
    )

    return candidates