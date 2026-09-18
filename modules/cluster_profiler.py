"""
Cluster profiling for TaskAware.

Builds interpretable profiles for each cluster and identifies
the features that differ most between clusters.

IMPORTANT: profiles describe observed differences in the
data. They are not ground-truth labels.
"""

import numpy as np
import pandas as pd


def build_cluster_profiles(
    df,
    feature_columns,
    labels,
    n_clusters,
    text_columns=None,
    representative_terms=None,
):
    """
    Build interpretable profiles for each cluster.

    Numerical features: mean and median.
    Categorical features: dominant category and ratio.
    Text columns: representative terms (if provided).
    """

    text_columns = text_columns or []

    if df is None or len(df) == 0:
        return []

    working = df[feature_columns].copy()

    working["__cluster__"] = labels

    profiles = []

    for cluster in range(n_clusters):

        subset = working[
            working["__cluster__"] == cluster
        ]

        if len(subset) == 0:
            continue

        features = []

        for column in feature_columns:

            series = subset[column].dropna()

            if len(series) == 0:
                continue

            if pd.api.types.is_numeric_dtype(series):

                features.append({
                    "column": column,
                    "type": "numerical",
                    "mean": round(
                        float(series.mean()),
                        3,
                    ),
                    "median": round(
                        float(series.median()),
                        3,
                    ),
                })

            else:

                counts = series.value_counts()

                dominant = counts.index[0]

                features.append({
                    "column": column,
                    "type": "categorical",
                    "dominant": str(dominant),
                    "dominant_ratio": round(
                        float(
                            counts.iloc[0] / len(series)
                        ),
                        3,
                    ),
                })

        profile = {
            "cluster": int(cluster),
            "size": int(len(subset)),
            "size_percentage": round(
                len(subset) / len(working) * 100,
                2,
            ),
            "features": features,
        }

        if representative_terms:
            profile["text_terms"] = representative_terms.get(
                cluster,
                [],
            )

        profiles.append(profile)

    return profiles


def find_distinguishing_features(
    profiles,
    feature_columns,
    limit=5,
):
    """
    Identify the features that differ most between clusters.
    """
    if len(profiles) < 2:
        return []

    differences = []

    for column in feature_columns:

        values = []

        for profile in profiles:

            for feature in profile["features"]:

                if feature["column"] != column:
                    continue

                if feature["type"] == "numerical":
                    values.append(feature["mean"])
                else:
                    values.append(feature["dominant"])

                break

        if len(values) < 2:
            continue

        if all(
            isinstance(v, (int, float))
            for v in values
        ):

            spread = max(values) - min(values)

            relative_spread = (
                spread / abs(np.mean(values))
                if abs(np.mean(values)) > 0
                else 0
            )

            differences.append({
                "column": column,
                "spread": round(spread, 3),
                "relative_spread": round(
                    float(relative_spread),
                    3,
                ),
                "cluster_values": values,
            })

    differences.sort(
        key=lambda item: item["relative_spread"],
        reverse=True,
    )

    return differences[:limit]


def representative_text_terms(
    X_transformed,
    labels,
    tfidf_pipeline,
    n_clusters,
    top_k=5,
    text_columns=None,
):
    """
    Find the highest-weight TF-IDF terms per cluster.

    X_transformed may be a dense or sparse TF-IDF matrix.
    tfidf_pipeline provides the term feature names.

    Returns a dict: { cluster: [terms] }
    """
    try:

        if X_transformed.shape[1] == 0:
            return {}

        feature_names = list(
            tfidf_pipeline.get_feature_names_out()
        )

        n_features = X_transformed.shape[1]

        # Only attempt when the feature count still aligns
        # with the original pipeline names (i.e. no SVD
        # reduction changed the feature space).
        if len(feature_names) != n_features:
            return {}

        if text_columns is None:
            text_columns = []

        result = {}

        for cluster in range(n_clusters):

            mask = (labels == cluster)

            if mask.sum() == 0:
                continue

            cluster_matrix = X_transformed[mask]

            if hasattr(cluster_matrix, "toarray"):
                mean_weights = np.asarray(
                    cluster_matrix.mean(axis=0)
                ).ravel()
            else:
                mean_weights = cluster_matrix.mean(axis=0)

            # Keep only terms from text source columns
            candidates = [
                (i, feature_names[i])
                for i in range(n_features)
                if ":" in feature_names[i]
                and feature_names[i].split(":", 1)[0]
                in text_columns
                and mean_weights[i] > 0
            ]

            candidates.sort(
                key=lambda item: mean_weights[item[0]],
                reverse=True,
            )

            result[cluster] = [
                name
                for _, name in candidates[:top_k]
            ]

        return result

    except Exception:
        return {}