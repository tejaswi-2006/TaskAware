import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    y_true,
    y_pred,
    classes=None
):
    """
    Compute classification metrics from true and
    predicted labels.

    Returns
    -------
    dict
        accuracy, precision, recall, f1, roc_auc (where
        applicable), class counts and confusion matrix.
    """

    labels = (
        classes
        if classes is not None
        else sorted(set(y_true.tolist()))
    )

    n_classes = len(labels)

    average = (
        "binary"
        if n_classes == 2
        else "macro"
    )

    precision = precision_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # ROC-AUC only makes sense for probabilistic models.
    # For 2-class problems it can be computed from scores,
    # so the caller provides them separately when possible.
    result = {
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)),
            4
        ),
        "precision": round(
            float(precision),
            4
        ),
        "recall": round(
            float(recall),
            4
        ),
        "f1": round(
            float(f1),
            4
        ),
        "roc_auc": None,
        "n_classes": n_classes,
        "labels": [str(label) for label in labels],
        "confusion_matrix": cm.tolist(),
        "class_counts": {
            str(label): int((y_true == label).sum())
            for label in labels
        }
    }

    return result


def roc_auc_from_scores(
    y_true,
    y_scores,
    classes=None
):
    """
    Compute ROC-AUC from predicted class probabilities.

    For binary problems this is straightforward.
    For multiclass problems it uses the one-vs-rest
    macro average.
    """

    labels = (
        classes
        if classes is not None
        else sorted(set(y_true.tolist()))
    )

    n_classes = len(labels)

    try:

        if n_classes == 2:

            return round(
                float(
                    roc_auc_score(
                        y_true,
                        y_scores[:, 1]
                    )
                ),
                4
            )

        if n_classes > 2:

            return round(
                float(
                    roc_auc_score(
                        y_true,
                        y_scores,
                        multi_class="ovr",
                        average="macro"
                    )
                ),
                4
            )

    except Exception:
        pass

    return None


# ============================================================
# REGRESSION METRICS
# ============================================================

def regression_metrics(
    y_true,
    y_pred
):
    """
    Compute regression metrics.

    Returns
    -------
    dict
        mae, mse, rmse, r2.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4)
    }