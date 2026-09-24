"""
Decision engine for TaskAware.

Decides the routing between:
    - supervised target validation
    - classification
    - regression
    - clustering (offered, not forced)

The decision engine uses the central schema analyzer and the
target evidence engine. It never assumes "no target means
clustering"; it offers clustering as an option.
"""

from modules.schema_analyzer import analyze_schema
from modules.objective_target import analyze_target_candidates


def decide_route(
    df,
    objective=None,
    selected_target=None,
    schema=None,
    candidates=None,
):
    """
    Determine the appropriate route for the dataset.

    Returns
    -------
    dict
        {
          "route": "supervised" | "clustering" | "ambiguous"
                   | "unsupported",
          "problem_type": "Classification" | "Regression"
                          | "Clustering" | None,
          "recommended_target": str | None,
          "candidates": [...],
          "target_role": str | None,
          "messages": [...],
          "needs_confirmation": bool,
          "group_confirmation_required": bool,
          "group_confirmation_status": str,
          "confirmed_group_column": None,
          "group_validation_strategy": str | None,
          "group_validation_reason": str | None,
        }
    """

    if schema is None:
        schema = analyze_schema(
            df,
            target_column=selected_target,
        )

    messages = []

    strategy = schema.get("validation_strategy", {})

    group_state = {
        "group_confirmation_required": strategy.get(
            "group_confirmation_required", False
        ),
        "group_confirmation_status": strategy.get(
            "group_confirmation_status", "not_required"
        ),
        "confirmed_group_column": None,
        "group_validation_strategy": strategy.get("strategy"),
        "group_validation_reason": strategy.get("reason"),
    }

    # ------------------------------------------------
    # UNSUPPORTED-ONLY DATA
    # ------------------------------------------------

    usable_roles = {
        "numeric",
        "categorical",
        "boolean",
        "text",
        "datetime",
    }

    has_usable = any(
        record["role"] in usable_roles
        for record in schema["columns"]
        if not record["is_target"]
    )

    if not has_usable:

        return {
            "route": "unsupported",
            "problem_type": None,
            "recommended_target": None,
            "candidates": [],
            "target_role": None,
            "messages": [
                "The dataset contains no usable feature "
                "columns. All columns were detected as "
                "identifiers, constants, or unsupported "
                "types."
            ],
            "needs_confirmation": True,
            **group_state,
        }

    # ------------------------------------------------
    # TARGET PROVIDED
    # ------------------------------------------------

    if selected_target and selected_target in df.columns:

        target_record = next(
            (
                record for record in schema["columns"]
                if record["column"] == selected_target
            ),
            None,
        )

        target_role = (
            target_record["role"]
            if target_record
            else None
        )

        # Unsupported target
        if target_role in ("identifier", "unsupported", "constant"):

            return {
                "route": "unsupported",
                "problem_type": None,
                "recommended_target": None,
                "candidates": [],
                "target_role": target_role,
                "messages": [
                    f"'{selected_target}' cannot be used as "
                    "a target because it was detected as an "
                    f"{target_role} column."
                ],
                "needs_confirmation": True,
                **group_state,
            }

        # Text target -> warn but allow with clear limitation
        if target_role == "text":

            return {
                "route": "supervised",
                "problem_type": None,
                "recommended_target": selected_target,
                "candidates": [],
                "target_role": target_role,
                "messages": [
                    "You selected a free-text column as the "
                    "target. The current prediction engine "
                    "supports structured categorical or "
                    "numerical targets, not free-text "
                    "prediction. Consider selecting a "
                    "categorical or numerical outcome "
                    "column instead."
                ],
                "needs_confirmation": True,
                **group_state,
            }

        # Datetime target
        if target_role == "datetime":

            return {
                "route": "supervised",
                "problem_type": None,
                "recommended_target": selected_target,
                "candidates": [],
                "target_role": target_role,
                "messages": [
                    "You selected a date column as the "
                    "target. TaskAware does not predict raw "
                    "dates. Select a structured outcome "
                    "instead."
                ],
                "needs_confirmation": True,
                **group_state,
            }

        # Valid target -> determine problem type
        problem_type = _problem_type_for_role(
            target_role,
            df[selected_target],
        )

        return {
            "route": "supervised",
            "problem_type": problem_type,
            "recommended_target": selected_target,
            "candidates": [],
            "target_role": target_role,
            "messages": [],
            "needs_confirmation": False,
            **group_state,
        }

    # ------------------------------------------------
    # NO TARGET PROVIDED
    # ------------------------------------------------

    if candidates is None:
        candidates = analyze_target_candidates(
            df,
            objective=objective,
            schema=schema,
        )

    usable = [
        c for c in candidates
        if c["suitability"]
        in ("Highly Suitable", "Suitable", "Possible")
        and not c["is_id"]
        and c["type"] not in ("text", "datetime")
    ]

    if not usable:

        return {
            "route": "clustering",
            "problem_type": "Clustering",
            "recommended_target": None,
            "candidates": [],
            "target_role": None,
            "messages": [
                "No suitable supervised target was found. "
                "TaskAware offers clustering to discover "
                "natural groups in the data."
            ],
            "needs_confirmation": True,
            **group_state,
        }

    recommended = usable[0]

    return {
        "route": "supervised",
        "problem_type": _problem_type_for_candidate(
            recommended
        ),
        "recommended_target": recommended["column"],
        "candidates": usable,
        "target_role": recommended["type"],
        "messages": [
            f"Recommended target: {recommended['column']}. "
            "Please confirm before proceeding."
        ],
        "needs_confirmation": True,
        **group_state,
    }


def _problem_type_for_role(target_role, series):
    """
    Determine classification vs regression from the target
    role and value distribution.
    """

    if target_role in ("categorical", "boolean"):
        return "Classification"

    if target_role == "numeric":

        unique_count = series.nunique(dropna=True)

        unique_ratio = (
            unique_count / len(series)
            if len(series) > 0
            else 0
        )

        if unique_count <= 10 and unique_ratio < 0.05:
            return "Classification"

        return "Regression"

    return None


def _problem_type_for_candidate(candidate):
    """
    Map a target candidate to a problem type.
    """

    mapping = {
        "Classification": "Classification",
        "Regression": "Regression",
    }

    return mapping.get(
        candidate.get("possible_problem_type"),
        None,
    )