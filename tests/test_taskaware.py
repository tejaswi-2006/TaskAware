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


def make_group_data():
    return pd.DataFrame({
        "account_id": [f"ACC-{i % 10:03d}" for i in range(200)],
        "device": ["TV", "Mobile", "Tablet", "Laptop", "Desktop"] * 40,
        "region": ["A", "B", "C", "D", "E", "F"] * 33 + ["A", "B"],
        "gender": ["M", "F", "X"] * 66 + ["M", "F"],
        "x": list(range(200)),
        "y": [0, 1] * 100,
    })


class TestSafeDuplicates(unittest.TestCase):

    def test_dict_column_partial(self):
        df = pd.DataFrame({
            "id": ["R0", "R1", "R2", "R0"],
            "payload": [{"a": 1}, {"b": [1, 2]}, {"a": 1}, {"a": 1}],
            "x": [1.0, 2.0, 3.0, 1.0],
        })
        validation = validate_dataset(df)
        self.assertEqual(
            validation["duplicate_check_status"], "partial"
        )
        self.assertFalse(validation["duplicate_check_complete"])
        self.assertIn(
            "payload",
            validation[
                "unsupported_columns_ignored_for_duplicate_check"
            ],
        )
        self.assertEqual(validation["duplicate_rows"], 1)
        self.assertIn("payload", validation["unsupported_columns"])

    def test_list_and_mixed_columns(self):
        df = pd.DataFrame({
            "tags": [["a"], ["b"], ["a"]],
            "y": [0, 1, 0],
        })
        validation = validate_dataset(df)
        self.assertEqual(
            validation["duplicate_check_status"], "partial"
        )
        self.assertEqual(validation["duplicate_rows"], 1)

    def test_normal_dataset_complete(self):
        df = pd.DataFrame({
            "a": [1, 2, 1, 3],
            "b": ["x", "y", "x", "z"],
        })
        validation = validate_dataset(df)
        self.assertEqual(
            validation["duplicate_check_status"], "complete"
        )
        self.assertTrue(validation["duplicate_check_complete"])
        self.assertEqual(validation["duplicate_rows"], 1)


class TestGroupSelection(unittest.TestCase):

    def test_ordinary_attributes_not_groups(self):
        df = make_group_data()
        schema = analyze_schema(df, target_column="y")

        for column in ("device", "region", "gender"):
            record = next(
                r for r in schema["columns"]
                if r["column"] == column
            )
            self.assertFalse(
                record["group_candidate"],
                column,
            )

        self.assertIn("account_id", schema["group_columns"])
        self.assertNotIn("device", schema["group_columns"])

    def test_unique_id_not_group(self):
        df = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(100)],
            "x": list(range(100)),
            "y": [0, 1] * 50,
        })
        schema = analyze_schema(df, target_column="y")
        self.assertIn("customer_id", schema["id_columns"])
        self.assertNotIn("customer_id", schema["group_columns"])

    def test_entity_ids_trigger_group_strategy(self):
        df = pd.DataFrame({
            "patient_id": [f"P{i % 12:03d}" for i in range(120)],
            "v": list(range(120)),
            "y": [0, 1] * 60,
        })
        schema = analyze_schema(df, target_column="y")
        self.assertIn("patient_id", schema["group_columns"])
        self.assertIn(
            "Group",
            schema["validation_strategy"]["strategy"],
        )
        self.assertEqual(
            schema["group_details"][0]["column"], "patient_id"
        )

    def test_group_split_keeps_entities_whole(self):
        df = make_group_data()
        training = train_models(df, "y", "Classification")
        self.assertEqual(
            training.get("used_group_split"), "account_id"
        )
        train_groups = set(
            df.loc[training["X_train"].index, "account_id"]
        )
        test_groups = set(
            df.loc[training["X_test"].index, "account_id"]
        )
        self.assertTrue(train_groups.isdisjoint(test_groups))


class TestNumericSemantics(unittest.TestCase):

    def test_tiered_measurement_stays_numeric(self):
        df = pd.DataFrame({
            "monthly_fee": [8.99, 13.99, 17.99] * 20,
            "number_of_profiles": [1, 2, 3, 4, 5] * 12,
            "y": [0, 1] * 30,
        })
        schema = analyze_schema(df, target_column="y")
        records = {
            record["column"]: record
            for record in schema["columns"]
        }
        self.assertEqual(
            records["monthly_fee"]["semantic_type"], "measurement"
        )
        self.assertEqual(
            records["monthly_fee"]["role"], "numeric"
        )
        self.assertEqual(
            records["number_of_profiles"]["semantic_type"],
            "count",
        )

    def test_semantic_alternative_present(self):
        df = pd.DataFrame({
            "monthly_fee": [8.99, 13.99, 17.99] * 20,
            "y": [0, 1] * 30,
        })
        schema = analyze_schema(df, target_column="y")
        record = next(
            r for r in schema["columns"]
            if r["column"] == "monthly_fee"
        )
        self.assertIn("semantic_alternative", record)
        self.assertIsNotNone(record["semantic_alternative"])

    def test_target_storage_and_semantic(self):
        from modules.problem_type_detector import (
            detect_problem_type,
        )

        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 60),
            "churned": [0, 1] * 30,
        })
        result = detect_problem_type(df, "churned")
        self.assertEqual(result["problem_type"], "Classification")
        self.assertEqual(result["storage_dtype"], "int64")
        self.assertEqual(result["semantic_type"], "binary_target")


class TestFeatureAccounting(unittest.TestCase):

    def test_prep_and_training_agree(self):
        df = pd.DataFrame({
            "id": [f"U{i}" for i in range(60)],
            "age": np.random.normal(40, 10, 60),
            "gender": ["M", "F"] * 30,
            "y": [0, 1] * 30,
        })
        prep = prepare_preprocessing(df, "y", "Classification")
        accounting = prep["feature_accounting"]

        self.assertEqual(accounting["raw_columns_total"], 4)
        self.assertEqual(accounting["raw_feature_columns"], 3)
        self.assertEqual(
            accounting["identifier_columns_removed"], ["id"]
        )
        self.assertEqual(accounting["raw_usable_count"], 2)
        self.assertFalse(accounting["feature_selection_applied"])
        self.assertEqual(
            accounting["final_model_features"],
            accounting["generated_transformed_features"],
        )

        training = train_models(df, "y", "Classification")
        self.assertEqual(
            training["n_transformed_features"],
            accounting["generated_transformed_features"],
        )
        self.assertEqual(
            training["feature_accounting"]["final_model_features"],
            accounting["final_model_features"],
        )


class TestReportConsistency(unittest.TestCase):

    def test_report_matches_pipeline(self):
        from modules.report_generator import collect_report_data

        df = pd.DataFrame({
            "account_id": [f"A{i % 8:02d}" for i in range(160)],
            "x1": np.random.normal(0, 1, 160),
            "x2": np.random.normal(5, 2, 160),
            "y": np.random.choice([0, 1], 160),
        })

        data = collect_report_data(
            df, "check.csv", "Predict y", "y", "Classification",
            "test explanation",
        )

        self.assertEqual(data["n_rows"], 160)
        self.assertEqual(data["target_column"], "y")
        self.assertEqual(data["problem_type"], "Classification")

        group = data["group_validation"]
        self.assertEqual(group["group_column"], "account_id")
        self.assertIn("Group", group["strategy"])
        self.assertIsNotNone(group["reason"])
        self.assertEqual(group["confidence"], "high")

        training = train_models(df, "y", "Classification")
        self.assertEqual(
            [m["name"] for m in data["model_results"]["results"]],
            [m["name"] for m in training["results"]],
        )
        self.assertEqual(
            data["model_results"]["n_transformed_features"],
            training["n_transformed_features"],
        )

        # Method honesty: report must carry the actual
        # explanation-layer method, never an assumed SHAP.
        from modules.model_trainer import train_named_model
        from modules.explainability import perform_explainability

        best = data["model_results"]["best_model"]
        fitted = train_named_model(
            df, "y", "Classification", best["name"]
        )
        actual = perform_explainability(
            df, "y", "Classification",
            pipeline=fitted["pipeline"],
            X=fitted["X_test"],
            y=fitted["y_test"],
        )
        self.assertEqual(
            data["model_results"]["explanation_method"],
            actual.get("method"),
        )


class TestAnalyticsUI(unittest.TestCase):

    TEMPLATES_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "templates",
    )

    FLOW_TEMPLATES = [
        "dataset_overview.html",
        "objective_analysis.html",
        "target_comparison.html",
        "target_confirmed.html",
        "detailed_eda.html",
        "preprocessing.html",
        "model_results.html",
        "clustering_results.html",
        "explainability.html",
        "ai_explanation.html",
        "report.html",
        "upload.html",
    ]

    def _read(self, name):
        with open(
            os.path.join(self.TEMPLATES_DIR, name),
            encoding="utf-8",
        ) as handle:
            return handle.read()

    def test_shared_assets_exist(self):
        base = os.path.dirname(os.path.dirname(__file__))
        self.assertTrue(
            os.path.exists(
                os.path.join(base, "static", "css", "taskaware.css")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(base, "static", "js", "taskaware.js")
            )
        )

    def test_css_accessibility_rules(self):
        base = os.path.dirname(os.path.dirname(__file__))

        with open(
            os.path.join(base, "static", "css", "taskaware.css"),
            encoding="utf-8",
        ) as handle:
            css = handle.read()

        self.assertIn("prefers-reduced-motion", css)
        self.assertIn('data-view="simple"', css)
        self.assertIn("focus-visible", css)
        self.assertIn("ta-skip-link", css)
        self.assertIn("@media", css)

    def test_js_toggles_without_webgl(self):
        base = os.path.dirname(os.path.dirname(__file__))

        with open(
            os.path.join(base, "static", "js", "taskaware.js"),
            encoding="utf-8",
        ) as handle:
            js = handle.read().lower()

        self.assertIn("ta-motion-toggle", js)
        self.assertIn("ta-simple-toggle", js)
        self.assertNotIn("three", js)
        self.assertNotIn("getcontext", js)
        self.assertNotIn("webglrenderingcontext", js)
        self.assertNotIn("<canvas", js)

    def test_no_webgl_dependency_in_templates(self):
        for name in os.listdir(self.TEMPLATES_DIR):
            if not name.endswith(".html"):
                continue

            with open(
                os.path.join(self.TEMPLATES_DIR, name),
                encoding="utf-8",
            ) as handle:
                html = handle.read().lower()

            self.assertNotIn("three.js", html, name)
            self.assertNotIn("<canvas", html, name)

    def test_flow_templates_share_chrome(self):
        for name in self.FLOW_TEMPLATES:
            html = self._read(name)
            themed = (
                "taskaware.css" in html
                or 'extends "base.html"' in html
            )
            self.assertTrue(themed, name)
            self.assertIn("ta-skip-link", html, name)
            has_toggles = (
                "_toggles.html" in html
                or 'extends "base.html"' in html
            )
            self.assertTrue(has_toggles, name)

    def test_timeline_renders_states(self):
        from app import app as flask_app

        with flask_app.test_request_context():
            from flask import render_template

            html = render_template(
                "_timeline.html",
                workflow_stage="target",
            )

        for stage in ["Upload", "Schema", "Target", "EDA",
                      "Preprocessing", "Modelling",
                      "Explainability", "Report"]:
            self.assertIn(stage, html)

        self.assertIn('aria-current="step"', html)
        self.assertIn("(completed)", html)
        self.assertIn("(current)", html)
        self.assertIn("(pending)", html)

    def test_report_method_branches(self):
        html = self._read("report.html")
        self.assertIn("not SHAP", html)
        self.assertIn("_method", html)
        self.assertIn("Group Validation", html)
        self.assertIn("Feature Accounting", html)
        self.assertIn("Reproducibility", html)

    def test_model_results_shows_per_class(self):
        html = self._read("model_results.html")
        self.assertIn("Per-Class Metrics", html)
        self.assertIn("imbalanc", html.lower())
        self.assertIn("Primary selection metric", html)
        self.assertIn("Train (s)", html)

    def test_overview_group_confirmation(self):
        html = self._read("dataset_overview.html")
        self.assertIn("/confirm-groups", html)
        self.assertIn("Confirmation Needed", html)
        self.assertIn("Continue Without Group Splitting", html)

    def test_per_class_metrics_backend(self):
        from modules.model_evaluator import (
            classification_metrics,
        )

        y_true = pd.Series([0] * 90 + [1] * 10)
        y_pred = pd.Series([0] * 85 + [1] * 5 + [0] * 8 + [1] * 2)
        metrics = classification_metrics(
            y_true, y_pred, classes=[0, 1]
        )

        self.assertEqual(len(metrics["per_class"]), 2)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIsNotNone(metrics["imbalance_warning"])

    def test_decision_group_keys(self):
        from modules.decision_engine import decide_route

        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 60),
            "y": [0, 1] * 30,
        })
        decision = decide_route(df, objective="Predict y")

        for key in ("group_confirmation_required",
                    "group_confirmation_status",
                    "confirmed_group_column",
                    "group_validation_strategy",
                    "group_validation_reason"):
            self.assertIn(key, decision)

    def test_confirm_groups_route(self):
        from app import app as flask_app

        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "uploads",
            "ui_confirm_check.csv",
        )

        pd.DataFrame({
            "post_id": [f"P{i % 12:03d}" for i in range(120)],
            "x": list(range(120)),
            "y": [0, 1] * 60,
        }).to_csv(path, index=False)

        try:
            client = flask_app.test_client()

            response = client.post(
                "/confirm-groups",
                data={"filename": "ui_confirm_check.csv"},
            )
            body = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Confirmation Needed", body)
            self.assertNotIn("Traceback", body)

            response = client.post(
                "/confirm-groups",
                data={"filename": "ui_confirm_check.csv",
                      "groups": "post_id"},
            )
            body = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn("post_id", body)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestRFTrainingConfig(unittest.TestCase):

    ENV_VARS = (
        "TASKAWARE_RF_TREES",
        "TASKAWARE_RF_JOBS",
        "TASKAWARE_SLOW_MODEL_SECONDS",
    )

    def setUp(self):
        from modules import model_recommender

        self._saved = {
            key: os.environ.get(key) for key in self.ENV_VARS
        }

        for key in self.ENV_VARS:
            os.environ.pop(key, None)

        import importlib
        importlib.reload(model_recommender)
        self.recommender = model_recommender

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        import importlib
        from modules import model_recommender
        importlib.reload(model_recommender)

    def test_defaults_retained(self):
        config = self.recommender.resolve_rf_config()
        self.assertEqual(config["n_estimators"], 100)
        self.assertEqual(config["n_jobs"], -1)
        self.assertEqual(config["warnings"], [])
        self.assertFalse(config["overridden"])

    def test_environment_overrides(self):
        os.environ["TASKAWARE_RF_TREES"] = "50"
        os.environ["TASKAWARE_RF_JOBS"] = "2"
        config = self.recommender.resolve_rf_config()
        self.assertEqual(config["n_estimators"], 50)
        self.assertEqual(config["n_jobs"], 2)
        self.assertTrue(config["overridden"])

    def test_invalid_values_fall_back_visibly(self):
        for var, bad in [
            ("TASKAWARE_RF_TREES", "0"),
            ("TASKAWARE_RF_TREES", "-3"),
            ("TASKAWARE_RF_TREES", "abc"),
            ("TASKAWARE_RF_TREES", "10.5"),
            ("TASKAWARE_RF_JOBS", "0"),
            ("TASKAWARE_RF_JOBS", "-5"),
            ("TASKAWARE_RF_JOBS", "many"),
            ("TASKAWARE_RF_JOBS", "2.0"),
        ]:
            with self.subTest(var=var, value=bad):
                os.environ.pop("TASKAWARE_RF_TREES", None)
                os.environ.pop("TASKAWARE_RF_JOBS", None)
                os.environ[var] = bad
                config = self.recommender.resolve_rf_config()

                if var == "TASKAWARE_RF_TREES":
                    self.assertEqual(config["n_estimators"], 100)
                else:
                    self.assertEqual(config["n_jobs"], -1)

                self.assertTrue(
                    any(var in warning
                        for warning in config["warnings"])
                )

    def test_empty_string_means_unset(self):
        # An explicitly empty variable behaves like an
        # unset one: silent defaults, no warning.
        os.environ["TASKAWARE_RF_TREES"] = ""
        os.environ["TASKAWARE_RF_JOBS"] = "   "
        config = self.recommender.resolve_rf_config()
        self.assertEqual(config["n_estimators"], 100)
        self.assertEqual(config["n_jobs"], -1)
        self.assertEqual(config["warnings"], [])

    def test_override_reaches_estimator(self):
        os.environ["TASKAWARE_RF_TREES"] = "17"
        schema = analyze_schema(
            pd.DataFrame({
                "x": np.random.normal(0, 1, 40),
                "y": [0, 1] * 20,
            }),
            target_column="y",
        )
        from modules.model_recommender import recommend_models

        models = recommend_models(schema, "Classification")
        forest = next(
            m for m in models if m["name"] == "Random Forest"
        )
        params = forest["estimator"].get_params()
        self.assertEqual(params["n_estimators"], 17)

    def test_timing_fields_present(self):
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 120),
            "y": [0, 1] * 60,
        })
        training = train_models(df, "y", "Classification")

        self.assertIn("preprocessing_seconds", training)
        self.assertIsInstance(
            training["preprocessing_seconds"], float
        )

        for result in training["results"]:
            self.assertIn("cv_seconds", result)
            self.assertIn("train_seconds", result)
            self.assertIn("predict_seconds", result)
            self.assertIn("warning", result)

    def test_rf_warning_behavior(self):
        from modules.model_trainer import rf_slow_warning

        warning = rf_slow_warning(
            "Random Forest", 5000, 35, 100, -1, 5,
            fit_seconds=1.0, cv_seconds=400.0,
            threshold_seconds=300.0,
        )
        self.assertIsNotNone(warning)

        for token in ["Random Forest", "5000", "100", "5",
                      "-1"]:
            self.assertIn(token, warning)

        self.assertIsNone(
            rf_slow_warning(
                "Random Forest", 5000, 35, 100, -1, 5,
                fit_seconds=1.0, cv_seconds=4.0,
                threshold_seconds=300.0,
            )
        )
        self.assertIsNone(
            rf_slow_warning(
                "Logistic Regression", 5000, 35, None,
                None, 5, fit_seconds=999.0,
                cv_seconds=999.0,
                threshold_seconds=300.0,
            )
        )

    def test_successful_rf_evaluation(self):
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 120),
            "y": [0, 1] * 60,
        })
        training = train_models(df, "y", "Classification")
        forest = next(
            r for r in training["results"]
            if r["name"] == "Random Forest"
        )
        self.assertIsNotNone(forest["cv_mean"])
        self.assertIsNone(forest["warning"])
        self.assertGreaterEqual(forest["cv_seconds"], 0)

    def test_failed_cv_reports_error(self):
        from sklearn.base import BaseEstimator, ClassifierMixin

        from modules.model_trainer import cross_validate_model
        from modules.preprocessing import (
            build_preprocessing_pipeline,
        )

        class _AlwaysFails(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                raise RuntimeError("synthetic fit failure")

            def predict(self, X):
                raise RuntimeError("synthetic fit failure")

        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 60),
        })
        y = pd.Series([0, 1] * 30)
        schema = analyze_schema(df.assign(y=y), target_column="y")
        preprocessor = build_preprocessing_pipeline(
            schema["numerical_columns"],
            schema["categorical_columns"],
            schema["text_columns"],
            boolean_columns=schema["boolean_columns"],
            date_columns=schema["date_columns"],
        )
        result = cross_validate_model(
            preprocessor,
            df,
            y,
            {"name": "Broken", "estimator": _AlwaysFails()},
            "Classification",
        )
        self.assertIsNone(result["mean"])
        self.assertTrue(result["error"])
        self.assertGreaterEqual(result["cv_seconds"], 0)

    def test_backward_compatible_keys(self):
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 120),
            "y": [0, 1] * 60,
        })
        training = train_models(df, "y", "Classification")

        for key in ("problem_type", "target_column",
                    "target_info", "schema", "X_train_shape",
                    "X_test_shape", "results", "models",
                    "used_chronological_split",
                    "used_group_split",
                    "n_transformed_features",
                    "feature_accounting"):
            self.assertIn(key, training)

        for result in training["results"]:
            for key in ("name", "cv_mean", "cv_std",
                        "cv_folds", "compatible_reason",
                        "accuracy", "f1"):
                self.assertIn(key, result)

        mapping = training["target_info"]["mapping"]
        self.assertTrue(
            all(isinstance(k, str) for k in mapping)
        )


def make_missing_target_data(n=200, missing=2, regression=False):
    rng = np.random.default_rng(4)

    if regression:
        target = rng.normal(50, 10, n)
    else:
        target = rng.choice([0, 1], n).astype(float)

    target[[3, 17][:missing]] = np.nan

    return pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "hours": rng.uniform(0, 30, n).round(2),
        "label": target,
    })


class TestMissingTarget(unittest.TestCase):

    def test_classification_missing_labels(self):
        df = make_missing_target_data()
        training = train_models(df, "label", "Classification")
        row_info = training["row_info"]

        self.assertEqual(row_info["rows_uploaded"], 200)
        self.assertEqual(
            row_info["rows_excluded_missing_target"], 2
        )
        self.assertEqual(
            row_info["rows_available_for_modeling"], 198
        )
        self.assertEqual(row_info["target_missing_count"], 2)
        self.assertIn("were excluded", row_info["target_warning"])
        self.assertIn("not imputed", row_info["target_warning"])
        self.assertEqual(
            training["X_train_shape"][0]
            + training["X_test_shape"][0],
            198,
        )
        self.assertEqual(int(training["y_train"].isnull().sum()), 0)
        self.assertEqual(int(training["y_test"].isnull().sum()), 0)
        self.assertTrue(
            all(m["cv_mean"] is not None
                for m in training["results"])
        )

    def test_regression_missing_labels(self):
        df = make_missing_target_data(
            n=120, missing=2, regression=True
        )
        training = train_models(df, "label", "Regression")
        row_info = training["row_info"]

        self.assertEqual(row_info["rows_excluded_missing_target"], 2)
        self.assertEqual(
            row_info["rows_available_for_modeling"], 118
        )
        self.assertTrue(
            all(m.get("rmse") is not None
                for m in training["results"])
        )

    def test_no_missing_target(self):
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 80),
            "y": [0, 1] * 40,
        })
        training = train_models(df, "y", "Classification")
        row_info = training["row_info"]

        self.assertEqual(row_info["rows_excluded_missing_target"], 0)
        self.assertIsNone(row_info["target_warning"])
        self.assertEqual(
            row_info["rows_available_for_modeling"], 80
        )

    def test_all_target_missing_fails_clearly(self):
        from modules.model_trainer import TaskAwareTrainingError

        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 30),
            "y": [np.nan] * 30,
        })

        with self.assertRaises(TaskAwareTrainingError):
            train_models(df, "y", "Classification")

    def test_no_nan_reaches_model_fit(self):
        from sklearn.base import BaseEstimator, ClassifierMixin
        from modules.model_trainer import train_and_evaluate
        from modules.preprocessing import (
            build_preprocessing_pipeline,
        )

        seen = {}

        class _Recorder(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                seen["y"] = pd.Series(y)
                from sklearn.tree import DecisionTreeClassifier
                self._inner = DecisionTreeClassifier(
                    random_state=42
                )
                self._inner.fit(X, y)
                return self

            def predict(self, X):
                return self._inner.predict(X)

        df = make_missing_target_data(n=120, missing=2)
        training = train_models(df, "label", "Classification")
        schema = training["schema"]
        preprocessor = build_preprocessing_pipeline(
            schema["numerical_columns"],
            schema["categorical_columns"],
            schema["text_columns"],
            boolean_columns=schema["boolean_columns"],
            date_columns=schema["date_columns"],
        )
        train_and_evaluate(
            training["X_train"],
            training["y_train"],
            training["X_test"],
            training["y_test"],
            {"name": "Recorder", "estimator": _Recorder(),
             "supports_probability": False},
            "Classification",
            n_classes=2,
            preprocessor=preprocessor,
        )
        self.assertEqual(int(seen["y"].isnull().sum()), 0)

    def test_chronological_split_labels(self):
        df = pd.DataFrame({
            "day": pd.date_range("2023-01-01", periods=60,
                                 freq="D").astype(str),
            "x": list(range(60)),
            "y": [0] * 30 + [1] * 30,
        })
        training = train_models(df, "y", "Classification")

        self.assertTrue(training["used_chronological_split"])
        # Test labels must come from the latest dates
        # (all class 1 here), never repeat training rows.
        self.assertEqual(training["y_test"].tolist(), [1] * 12)
        self.assertTrue(
            set(training["X_test"].index).isdisjoint(
                set(training["X_train"].index)
            )
        )
        self.assertEqual(
            len(training["X_train"]) + len(training["X_test"]),
            60,
        )

    def test_group_length_mismatch_fails(self):
        from modules.model_trainer import (
            _split_data,
            TrainingValidationError,
        )

        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 40),
            "y": [0, 1] * 20,
        })
        schema = analyze_schema(df, target_column="y")
        X = df[["x"]]
        y = pd.Series([0, 1] * 20)

        with self.assertRaises(TrainingValidationError) as ctx:
            _split_data(
                X, y, "Classification", schema,
                groups=pd.Series(["a", "b"] * 10),
                group_column="g",
            )

        self.assertEqual(
            ctx.exception.details["error_code"],
            "MISALIGNED_GROUPS",
        )

    def test_prep_row_counts_consistent(self):
        from modules.preprocessing import prepare_preprocessing

        df = make_missing_target_data(n=100, missing=2)
        prep = prepare_preprocessing(
            df, "label", "Classification"
        )

        self.assertEqual(prep["n_rows"], 98)
        self.assertEqual(prep["rows_uploaded"], 100)
        self.assertEqual(
            prep["row_info"]["rows_excluded_missing_target"], 2
        )
        self.assertIn(
            "missing target rows excluded",
            " ".join(prep["steps"]),
        )

    def test_expansion_warning(self):
        from modules.preprocessing import prepare_preprocessing

        df = pd.DataFrame({
            "zip": [f"Z{i % 600:04d}" for i in range(1200)],
            "y": [0, 1] * 600,
        })
        prep = prepare_preprocessing(df, "y", "Classification")

        self.assertIsNotNone(
            prep["preview"]["expansion_warning"]
        )
        self.assertIsNotNone(
            prep["feature_accounting"]["expansion_warning"]
        )

        small = pd.DataFrame({
            "zip": ["a", "b", "c"] * 20,
            "y": [0, 1] * 30,
        })
        prep_small = prepare_preprocessing(
            small, "y", "Classification"
        )
        self.assertIsNone(
            prep_small["preview"]["expansion_warning"]
        )

    def test_api_fields_preserved(self):
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 80),
            "y": [0, 1] * 40,
        })
        training = train_models(df, "y", "Classification")

        for key in ("problem_type", "target_column",
                    "target_info", "schema", "X_train_shape",
                    "X_test_shape", "results", "models",
                    "used_chronological_split",
                    "used_group_split",
                    "n_transformed_features",
                    "feature_accounting", "row_info"):
            self.assertIn(key, training)

        row_info = training["row_info"]

        for key in ("rows_uploaded",
                    "rows_available_for_modeling",
                    "rows_excluded_missing_target",
                    "target_missing_count",
                    "target_warning"):
            self.assertIn(key, row_info)


class TestPremiumUI(unittest.TestCase):

    TEMPLATES_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "templates",
    )

    def _read(self, name):
        with open(
            os.path.join(self.TEMPLATES_DIR, name),
            encoding="utf-8",
        ) as handle:
            return handle.read()

    def test_base_shell_exists(self):
        html = self._read("base.html")
        self.assertIn("ta-sidebar", html)
        self.assertIn("ta-appbar", html)
        self.assertIn("ta-main-col", html)
        self.assertIn("block content", html)
        self.assertIn("aria-label", html)

    def test_light_theme_with_dark_mode(self):
        base = os.path.dirname(os.path.dirname(__file__))

        with open(
            os.path.join(base, "static", "css", "taskaware.css"),
            encoding="utf-8",
        ) as handle:
            css = handle.read()

        self.assertIn("--ta-bg: #f5f7fb", css)
        self.assertIn('html[data-theme="dark"]', css)
        self.assertIn("ta-sidebar", css)
        self.assertIn("ta-hero", css)
        self.assertIn("ta-kpi", css)

    def test_theme_toggle_persists(self):
        base = os.path.dirname(os.path.dirname(__file__))

        with open(
            os.path.join(base, "static", "js", "taskaware.js"),
            encoding="utf-8",
        ) as handle:
            js = handle.read()

        self.assertIn("ta-theme", js)
        self.assertIn("localStorage", js)
        # Animated counters must respect reduced motion.
        self.assertIn("motionReduced", js)

    def test_landing_content(self):
        html = self._read("index.html")
        self.assertIn("TASKAWARE", html)
        self.assertIn(
            "From Raw Data to Explainable Intelligence", html
        )
        self.assertIn("Start New Analysis", html)
        self.assertIn("Explore TaskAware", html)
        self.assertIn("ta-pipeline-3d", html)
        for capability in ("Dataset Intelligence",
                           "Target Intelligence",
                           "Automated Preprocessing",
                           "Model Recommendation",
                           "Explainable AI",
                           "Knowledge Reuse"):
            self.assertIn(capability, html)

    def test_landing_route(self):
        from app import app as flask_app

        response = flask_app.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("TASKAWARE", body)
        self.assertIn("Start New Analysis", body)
        self.assertNotIn("Traceback", body)

    def test_dashboard_honest_counts(self):
        # Dashboard must only show real counts, never
        # invented model metrics.
        html = self._read("dashboard.html")
        self.assertIn("data-count-up", html)
        self.assertIn("Recent Analyses", html)
        self.assertNotIn("92.4", html)
        self.assertNotIn("BEST F1", html)

    def test_dashboard_route(self):
        from app import app as flask_app

        client = flask_app.test_client()

        with client.session_transaction() as session:
            session["user_id"] = 1
            session["username"] = "tester"

        response = client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Recent Analyses", body)
        self.assertIn("ta-sidebar", body)
        self.assertIn("ta-appbar", body)
        self.assertNotIn("Traceback", body)

    def test_sidebar_states_not_color_only(self):
        html = self._read("base.html")
        # Every nav state carries text or an icon.
        self.assertIn("ta-side-icon", html)
        self.assertIn("(current)", html)
        self.assertIn("(completed)", html)
        self.assertIn("(pending)", html)


if __name__ == "__main__":
    unittest.main()