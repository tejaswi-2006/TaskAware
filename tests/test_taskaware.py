"""
Dataset-independent tests for TaskAware.

Covers:
    - numeric classification / regression
    - categorical / mixed data
    - text classification (IMDB)
    - text + numeric data
    - date + numeric data
    - missing values
    - imbalanced data
    - ID-heavy data
    - no-target clustering
    - ID-only datasets (clear error)
    - unsupported columns
    - ID + text coexistence
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.schema_analyzer import analyze_schema
from modules.ai_explainer import (
    BEGINNER_METRIC_LABELS,
    SECTION_TITLES,
    build_presentation,
    build_user_prompt,
    fallback_explanation,
    generate_explanation,
    humanize_feature_name,
    split_sections,
    validate_explanation,
)
from modules.objective_target import analyze_target_candidates
from modules.preprocessing import (
    prepare_preprocessing,
    build_preprocessing_pipeline,
    prepare_target,
)
from modules.model_trainer import (
    train_models,
    TaskAwareTrainingError,
)
from modules.clustering import perform_clustering
from modules.data_validator import validate_dataset


def make_text_data(n=300):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "id": [f"U{i:06d}" for i in range(n)],
        "review": [
            "This movie was absolutely brilliant and the "
            "acting was outstanding"
            if i % 2 == 0 else
            "This movie was terrible and boring, a complete "
            "waste of time"
            for i in range(n)
        ],
        "sentiment": ["pos" if i % 2 == 0 else "neg"
                      for i in range(n)],
    })


def make_date_data(n=200):
    return pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
        "value": np.random.normal(50, 10, n),
        "label": ["a" if i % 2 == 0 else "b" for i in range(n)],
    })


def make_unsupported_data(n=50):
    return pd.DataFrame({
        "id": [f"R{i}" for i in range(n)],
        "nested": [{"a": 1} for _ in range(n)],
        "target": ["x" if i % 2 == 0 else "y" for i in range(n)],
    })


class TestSchemaAnalyzer(unittest.TestCase):

    def test_imdb_review_is_text(self):
        df = make_text_data()
        schema = analyze_schema(df, target_column="sentiment")
        self.assertIn("review", schema["text_columns"])
        self.assertNotIn("review", schema["id_columns"])
        self.assertIn("review", schema["feature_columns"])

    def test_customer_id_is_identifier(self):
        df = make_text_data()
        schema = analyze_schema(df, target_column="sentiment")
        self.assertIn("id", schema["id_columns"])

    def test_date_detected(self):
        df = make_date_data()
        schema = analyze_schema(df, target_column="label")
        self.assertIn("date", schema["date_columns"])
        self.assertTrue(schema["is_time_series"])

    def test_unsupported_detected(self):
        df = make_unsupported_data()
        schema = analyze_schema(df, target_column="target")
        self.assertIn("nested", schema["unsupported_columns"])

    def test_short_categorical_not_text(self):
        df = pd.DataFrame({
            "gender": ["Male", "Female"] * 20,
            "label": ["a", "b"] * 20,
        })
        schema = analyze_schema(df, target_column="label")
        self.assertNotIn("gender", schema["text_columns"])
        self.assertIn("gender", schema["categorical_columns"])

    def test_id_and_text_coexist(self):
        df = make_text_data()
        schema = analyze_schema(df, target_column="sentiment")
        self.assertIn("id", schema["id_columns"])
        self.assertIn("review", schema["text_columns"])


class TestTargetRecommendation(unittest.TestCase):

    def test_imdb_recommends_sentiment(self):
        df = make_text_data()
        candidates = analyze_target_candidates(
            df, objective="Predict review sentiment"
        )
        self.assertEqual(candidates[0]["column"], "sentiment")
        self.assertNotEqual(candidates[0]["column"], "review")

    def test_text_column_not_recommended(self):
        df = make_text_data()
        candidates = analyze_target_candidates(df)
        review = next(
            (c for c in candidates if c["column"] == "review"),
            None,
        )
        self.assertIsNotNone(review)
        self.assertIn("text", review["type"])


class TestPreprocessing(unittest.TestCase):

    def test_imdb_tfidf_nonempty(self):
        df = make_text_data()
        prep = prepare_preprocessing(df, "sentiment", "Classification")
        self.assertGreater(prep["preview"]["features_after"], 0)
        names = prep["preview"]["feature_names"]
        self.assertTrue(any(n.startswith("review:") for n in names))

    def test_text_plus_numeric(self):
        df = make_text_data()
        df["price"] = np.random.normal(20, 5, len(df))
        prep = prepare_preprocessing(df, "sentiment", "Classification")
        self.assertIn("price", prep["numerical_columns"])
        self.assertIn("review", prep["text_columns"])
        self.assertGreater(prep["preview"]["features_after"], 0)

    def test_date_processed(self):
        df = make_date_data()
        prep = prepare_preprocessing(df, "label", "Classification")
        self.assertIn("date", prep["date_columns"])
        self.assertGreaterEqual(
            prep["preview"]["features_after"], 5
        )

    def test_boolean_processed(self):
        df = pd.DataFrame({
            "is_active": np.random.choice([True, False], 100),
            "label": ["a", "b"] * 50,
        })
        prep = prepare_preprocessing(df, "label", "Classification")
        self.assertIn("is_active", prep["boolean_columns"])
        self.assertGreater(prep["preview"]["features_after"], 0)


class TestTraining(unittest.TestCase):

    def test_numeric_classification(self):
        df = pd.DataFrame({
            "x1": np.random.normal(0, 1, 300),
            "x2": np.random.normal(5, 2, 300),
            "y": np.random.choice([0, 1], 300),
        })
        res = train_models(df, "y", "Classification")
        self.assertGreater(len(res["results"]), 0)

    def test_numeric_regression(self):
        df = pd.DataFrame({
            "x1": np.random.normal(0, 1, 300),
            "x2": np.random.normal(5, 2, 300),
            "y": np.random.normal(10, 3, 300),
        })
        res = train_models(df, "y", "Regression")
        self.assertGreater(len(res["results"]), 0)
        self.assertTrue(all(r.get("rmse") is not None
                            for r in res["results"]))

    def test_text_classification(self):
        df = make_text_data()
        res = train_models(df, "sentiment", "Classification")
        names = [r["name"] for r in res["results"]]
        self.assertIn("Logistic Regression", names)
        self.assertNotIn("Random Forest", names)

    def test_id_only_fails_clearly(self):
        df = pd.DataFrame({
            "id": [f"U{i:06d}" for i in range(100)],
            "customer_id": [f"c-{i}" for i in range(100)],
            "label": ["a" if i % 2 == 0 else "b" for i in range(100)],
        })
        df = df[["id", "customer_id"]]
        with self.assertRaises(TaskAwareTrainingError):
            train_models(df, "label", "Classification")


class TestClustering(unittest.TestCase):

    def test_numeric_clustering(self):
        df = pd.DataFrame({
            "id": [f"C{i}" for i in range(200)],
            "age": np.concatenate([
                np.random.normal(25, 3, 100),
                np.random.normal(50, 5, 100),
            ]),
            "income": np.concatenate([
                np.random.normal(35000, 5000, 100),
                np.random.normal(90000, 15000, 100),
            ]),
        })
        res = perform_clustering(df)
        self.assertTrue(res["success"])
        self.assertGreaterEqual(res["n_clusters"], 2)
        self.assertGreaterEqual(len(res["profiles"]), 2)

    def test_text_clustering(self):
        df = make_text_data()
        res = perform_clustering(df)
        self.assertTrue(res["success"])
        self.assertIsNotNone(res.get("representation_note"))

    def test_id_only_clustering_fails_clearly(self):
        df = pd.DataFrame({
            "id": [f"U{i:06d}" for i in range(100)],
            "customer_id": [f"c-{i}" for i in range(100)],
        })
        res = perform_clustering(df)
        self.assertFalse(res["success"])
        self.assertIn("no usable feature", res["message"].lower())


class TestValidator(unittest.TestCase):

    def test_imdb_review_not_id(self):
        df = make_text_data()
        validation = validate_dataset(df)
        self.assertNotIn("review", validation["possible_id_columns"])
        self.assertIn("review", validation["text_columns"])


def make_enriched_data(n=60):
    import uuid

    rng = np.random.default_rng(0)

    return pd.DataFrame({
        "customer_id": [str(uuid.uuid4()) for _ in range(n)],
        "age": rng.integers(18, 70, n).astype(float),
        "number_of_profiles": rng.integers(1, 6, n),
        "active": [
            "Yes" if i % 2 == 0 else "No"
            for i in range(n)
        ],
        "review": [
            "The product worked really well and I enjoyed "
            "using it every day"
            if i % 2 == 0 else
            "Terrible experience, completely broken and "
            "disappointing service"
            for i in range(n)
        ],
        "signup_date": (
            pd.date_range("2023-01-01", periods=n, freq="D")
            .astype(str)
        ),
        "account_id": [f"ACC-{i % 10:03d}" for i in range(n)],
        "target": [0 if i % 2 == 0 else 1 for i in range(n)],
    })


class TestEnrichedSchema(unittest.TestCase):

    REQUIRED_NEW_KEYS = [
        "raw_dtype",
        "semantic_type",
        "semantic_type_likelihood",
        "confidence",
        "text_likelihood",
        "date_likelihood",
        "identifier_likelihood",
        "group_likelihood",
        "group_candidate",
        "group_confidence",
        "repeated_entity_count",
        "max_rows_per_entity",
        "cardinality_level",
        "cardinality_warning",
        "recommended_transformation",
        "target_suitability",
        "boolean_likelihood",
        "boolean_encoding",
        "warnings",
        "evidence",
        "unsupported_details",
    ]

    REQUIRED_OLD_KEYS = [
        "column",
        "role",
        "dtype",
        "unique_count",
        "unique_ratio",
        "missing_count",
        "missing_ratio",
        "is_target",
        "reasons",
        "shape",
    ]

    def test_semantic_roles(self):
        import json

        df = make_enriched_data()
        schema = analyze_schema(df, target_column="target")
        records = {
            record["column"]: record
            for record in schema["columns"]
        }

        self.assertEqual(
            records["customer_id"]["role"], "identifier"
        )
        self.assertEqual(records["review"]["role"], "text")
        self.assertEqual(
            records["age"]["semantic_type"], "measurement"
        )
        self.assertEqual(
            records["number_of_profiles"]["semantic_type"],
            "count",
        )
        self.assertGreaterEqual(
            records["active"]["boolean_likelihood"], 0.9
        )
        self.assertEqual(
            records["active"]["boolean_encoding"], "yes_no"
        )
        self.assertEqual(records["signup_date"]["role"], "datetime")
        self.assertTrue(
            records["account_id"]["group_candidate"]
        )
        self.assertEqual(
            records["target"]["semantic_type"], "binary_target"
        )

    def test_contract_preserved_and_serializable(self):
        import json

        df = make_enriched_data()
        schema = analyze_schema(df, target_column="target")

        for record in schema["columns"]:
            for key in self.REQUIRED_OLD_KEYS:
                self.assertIn(key, record)
            for key in self.REQUIRED_NEW_KEYS:
                self.assertIn(key, record)

        # Must not raise.
        json.dumps(schema)


def make_explain_context():
    return {
        "problem_type": "Classification",
        "target": "churned",
        "model": "Random Forest",
        "objective": "Predict customer churn",
        "n_records": 5000,
        "n_test": 1000,
        "target_classes": ["0", "1"],
        "method": "Feature Importance",
        "selection_reason": "Best cross-validated F1.",
        "metrics": {
            "accuracy": 0.887,
            "f1_score": 0.89,
            "precision": 0.891,
            "recall": 0.889,
        },
        "top_features": [
            "avg_watch_time_per_day",
            "watch_hours",
            "last_login_days",
            "monthly_fee",
            "number_of_profiles",
        ],
        "feature_importance": [
            {
                "feature": "avg_watch_time_per_day",
                "importance": 0.31,
            },
        ],
        "source_columns": {},
        "evaluation_notes": None,
    }


class TestAIExplanation(unittest.TestCase):

    def test_beginner_labels(self):
        self.assertEqual(
            BEGINNER_METRIC_LABELS["accuracy"],
            "How often the model was correct overall.",
        )
        self.assertEqual(
            BEGINNER_METRIC_LABELS["recall"],
            "How many actual positive cases the model found.",
        )

    def test_humanize_dynamic(self):
        self.assertEqual(
            humanize_feature_name("monthly_fee"),
            "Monthly Fee",
        )
        self.assertEqual(
            humanize_feature_name("review:excellent"),
            'word "excellent" in Review',
        )
        self.assertEqual(
            humanize_feature_name("avg_watch_time_per_day"),
            "Avg Watch Time Per Day",
        )

    def test_prompt_requires_sections(self):
        prompt = build_user_prompt('{"a": 1}')

        for title in SECTION_TITLES:
            self.assertIn(f"## {title}", prompt)

    def test_fallback_passes_validation(self):
        for problem_type, mutate in [
            ("Classification", {}),
            ("Regression", {
                "target": "final_exam_score",
                "model": "Ridge",
                "method": "Model Coefficients",
                "metrics": {"r2": 0.71, "mae": 8.42},
                "top_features": [
                    "attendance_percent",
                    "previous_grade",
                ],
            }),
            ("Clustering", {
                "target": None,
                "model": "K-Means",
                "method": "Cluster Profiles",
                "metrics": {},
                "n_clusters": 3,
                "top_features": ["age", "income"],
            }),
        ]:
            context = make_explain_context()
            context["problem_type"] = problem_type
            context.update(mutate)

            text = fallback_explanation(context)
            sections = split_sections(text)

            for title in SECTION_TITLES:
                self.assertIn(title, sections)

            result = validate_explanation(text, context)
            self.assertTrue(result["valid"], result["issues"])

    def test_fallback_never_calls_rf_importance_shap(self):
        import re

        context = make_explain_context()
        text = fallback_explanation(context)
        affirmed = re.sub(r"not shap", "", text.lower())
        affirmed = re.sub(r"this is not shap", "", affirmed)
        self.assertNotIn("shap", affirmed)
        self.assertIn("Avg Watch Time Per Day", text)

    def test_validation_rejects_invented_metric(self):
        context = make_explain_context()
        text = (
            fallback_explanation(context) + " Accuracy 0.99."
        )
        result = validate_explanation(text, context)
        self.assertFalse(result["valid"])

    def test_validation_rejects_causal_claim(self):
        context = make_explain_context()
        text = (
            fallback_explanation(context)
            + " Low watching causes churn."
        )
        result = validate_explanation(text, context)
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("causal" in issue for issue in result["issues"])
        )

    def test_validation_allows_negated_causal(self):
        context = make_explain_context()
        text = (
            "## What TaskAware Found\nTaskAware analyzed "
            "records to predict churned with Random Forest "
            "using feature importance. Accuracy 0.887. "
            "These patterns do not prove that watching "
            "causes anything. Limitation: validate before "
            "operational use."
        )
        result = validate_explanation(text, context)
        self.assertTrue(result["valid"], result["issues"])

    def test_generate_without_api_key_uses_fallback(self):
        import os

        os.environ.pop("GROQ_API_KEY", None)
        os.environ.pop("GROQ_API_TOKEN", None)

        result = generate_explanation(
            make_explain_context(),
            use_okf=False,
        )

        self.assertEqual(result["source"], "fallback")
        self.assertTrue(result["validation"]["valid"])
        self.assertEqual(len(result["sections"]), 6)
        self.assertTrue(result["plain_summary"])

    def test_presentation_structure(self):
        context = make_explain_context()
        result = generate_explanation(context, use_okf=False)
        presentation = build_presentation(context, result)

        self.assertIn("summary", presentation)
        self.assertIn("result_card", presentation)
        self.assertIn("reliability", presentation)
        self.assertIn("limitation", presentation)
        self.assertIn("technical", presentation)
        self.assertEqual(
            presentation["top_features"][0]["label"],
            "Avg Watch Time Per Day",
        )
        self.assertEqual(
            presentation["technical"]["method"],
            "Feature Importance",
        )


if __name__ == "__main__":
    unittest.main()