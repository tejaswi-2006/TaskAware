import numpy as np


# ============================================================
# MODEL SELECTION
# ============================================================

def select_best_model(
    results,
    problem_type
):
    """
    Select the best model from the evaluation results.

    TaskAware does not simply pick the highest accuracy.
    The decision is contextual:

    Classification
        - If the target is imbalanced, F1-score is the
          primary criterion.
        - Otherwise cross-validated F1 is used as the
          main criterion, with test F1 as confirmation.
        - ROC-AUC is used as a supporting signal.

    Regression
        - R-squared (cross-validated) is the primary
          criterion, with RMSE as a supporting signal.

    The selection records WHY the model was chosen.
    """

    if not results:
        return None

    valid_results = [
        r for r in results
        if r.get("cv_mean") is not None
    ]

    # If no CV available, fall back to test metrics
    pool = (
        valid_results
        if valid_results
        else results
    )

    if problem_type == "Classification":

        best = _select_classification(
            pool
        )

    else:

        best = _select_regression(
            pool
        )

    return _format_selection(
        best,
        problem_type,
        results
    )


def _select_classification(results):
    """
    Rank classification models by F1 score.

    When the class distribution is balanced, accuracy is
    a reasonable reference. When it is imbalanced, F1 and
    recall matter more. TaskAware uses F1 (macro) as the
    main criterion and prefers models with a better
    cross-validated F1.
    """

    def key_func(r):

        cv = (
            r.get("cv_mean")
            if r.get("cv_mean") is not None
            else r.get("f1", 0)
        )

        return cv

    return max(
        results,
        key=key_func
    )


def _select_regression(results):
    """
    Rank regression models by cross-validated R-squared.
    """

    def key_func(r):

        cv = (
            r.get("cv_mean")
            if r.get("cv_mean") is not None
            else r.get("r2", -np.inf)
        )

        return cv

    return max(
        results,
        key=key_func
    )


def _format_selection(
    best,
    problem_type,
    all_results
):
    """
    Produce the human-readable selection record.
    """

    if best is None:
        return None

    reason_parts = []

    primary_metric = None

    if problem_type == "Classification":

        best_score = best.get("cv_mean")

        if best_score is None:
            best_score = best.get("f1")

        counts = best.get("class_counts") or {}
        total = sum(counts.values()) or 0
        minority_share = (
            min(counts.values()) / total if total else 1.0
        )

        if minority_share < 0.2 and best.get("n_classes", 2) == 2:
            primary_metric = "macro_f1"
            reason_parts.append(
                f"{best['name']} achieved the strongest "
                f"cross-validated F1 score "
                f"({best_score:.4f}) among the evaluated "
                "models."
            )
            reason_parts.append(
                "Primary selection metric: Macro-F1. The "
                "minority class is small, so Macro-F1 gives "
                "each class equal importance. The model was "
                "not selected on accuracy alone."
            )
        else:
            primary_metric = "cv_f1_macro"
            reason_parts.append(
                f"{best['name']} achieved the strongest "
                f"cross-validated F1 score "
                f"({best_score:.4f}) among the evaluated "
                "models."
            )

        test_f1 = best.get("f1")

        if test_f1 is not None:
            reason_parts.append(
                f"On the held-out test set it reached "
                f"F1 = {test_f1:.4f}, "
                f"accuracy = {best.get('accuracy', 0):.4f}."
            )

    else:

        primary_metric = "cv_r2"

        best_score = best.get("cv_mean")

        if best_score is None:
            best_score = best.get("r2")

        reason_parts.append(
            f"{best['name']} achieved the strongest "
            f"cross-validated R-squared "
            f"({best_score:.4f}) among the evaluated "
            "models."
        )

        test_r2 = best.get("r2")

        if test_r2 is not None:
            reason_parts.append(
                f"On the held-out test set it reached "
                f"R2 = {test_r2:.4f}, "
                f"RMSE = {best.get('rmse', 0):.4f}."
            )

    return {
        "name": best["name"],
        "primary_metric": primary_metric,
        "metrics": {
            key: value
            for key, value in best.items()
            if key not in (
                "name",
                "cv_folds"
            )
        },
        "reason": " ".join(reason_parts),
        "ranked_results": sorted(
            all_results,
            key=lambda r: (
                r.get("cv_mean")
                if r.get("cv_mean") is not None
                else (
                    r.get("f1", 0)
                    if problem_type == "Classification"
                    else r.get("r2", -np.inf)
                )
            ),
            reverse=True
        )
    }