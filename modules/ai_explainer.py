"""
Beginner-friendly AI explanation layer for TaskAware.

Design
------
Progressive disclosure, without removing technical detail:

    Level 1 - plain-language narrative shown first.
    Level 2 - important metrics and model-selection details.
    Level 3 - technical details (SHAP, coefficients,
              transformed features, full tables) behind
              expandable sections.

Honesty rules (enforced by validate_explanation):

    - Never invent metrics or facts.
    - Never calculate metrics (Python computes everything).
    - Never claim a feature causes the prediction.
    - Never call Random Forest feature importance SHAP.
    - Never show raw feature names as the primary
      explanation (readable labels are used instead).
"""

import json
import os
import re

import requests


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "llama-3.3-70b-versatile"

GROQ_API_URL = (
    "https://api.groq.com/openai/v1/chat/completions"
)

MAX_TOP_FEATURES = 5


def get_api_key():
    """
    Read the Groq API key from environment variables.
    """
    return (
        os.environ.get("GROQ_API_KEY", "")
        or os.environ.get("GROQ_API_TOKEN", "")
    )


# ============================================================
# BEGINNER-FRIENDLY LABELS
# ============================================================

BEGINNER_METRIC_LABELS = {
    "accuracy": "How often the model was correct overall.",
    "precision": "How often a positive prediction was correct.",
    "recall": "How many actual positive cases the model found.",
    "f1_score": "A balance between precision and recall.",
    "f1": "A balance between precision and recall.",
    "r2": (
        "How much of the variation in the numbers the "
        "model explains (closer to 1 is better)."
    ),
    "mae": "The average size of the model's mistakes.",
    "rmse": (
        "The typical size of the model's mistakes, giving "
        "extra weight to large mistakes."
    ),
}

BEGINNER_TASK_LABELS = {
    "Classification": "Choosing between categories.",
    "Regression": "Predicting a numerical value.",
    "Clustering": "Finding groups of similar records.",
}

BEGINNER_CONCEPT_LABELS = {
    "feature importance": (
        "Which data fields helped the model most."
    ),
    "SHAP": (
        "How individual features contributed to a "
        "particular prediction."
    ),
}

SECTION_TITLES = [
    "What TaskAware Found",
    "Why the Model Was Selected",
    "What Influenced the Result",
    "How Reliable Is It?",
    "Important Limitation",
    "Simple Conclusion",
]

METHOD_PLAIN_NAMES = {
    "SHAP": "SHAP",
    "Model Coefficients": "model coefficients",
    "Feature Importance": "feature importance",
    "Permutation Importance": "permutation importance",
    "Cluster Profiles": (
        "cluster profiles, i.e. group averages "
        "(this is not SHAP)"
    ),
}


# ============================================================
# READABLE FEATURE NAMES (dynamic, any dataset)
# ============================================================

def humanize_feature_name(name, source_columns=None):
    """
    Translate a raw/transformed feature name into a readable
    description. Works for arbitrary datasets; nothing is
    hard-coded per dataset.

    Examples
    --------
    monthly_fee            -> Monthly Fee
    avg_watch_time_per_day -> Avg Watch Time Per Day
    review:excellent       -> word "excellent" in Review
    encoder__gender_Female -> Gender Female
    """
    raw = str(name).strip()

    source = None

    if isinstance(source_columns, dict):
        source = source_columns.get(raw)

    # TF-IDF style token: "review:excellent".
    if ":" in raw:
        prefix, token = raw.split(":", 1)
        prefix = prefix.strip()
        token = token.strip()

        if prefix and token:
            return (
                f'word "{token}" in '
                f"{humanize_feature_name(prefix)}"
            )

    # sklearn-style transformed name: drop the transformer
    # prefix, e.g. "onehot__subscription_type_Basic".
    if "__" in raw:
        raw = raw.split("__")[-1].strip()

    label = raw.replace("_", " ").replace("-", " ")
    label = re.sub(r"\s+", " ", label).strip()

    if not label:
        return str(name)

    words = []

    for word in label.split(" "):
        words.append(
            word[:1].upper() + word[1:]
            if word else word
        )

    label = " ".join(words)

    if source == "text":
        label = f"{label} (from text)"

    return label


def describe_problem(problem_type):
    """
    Beginner-friendly description of the ML task.
    """
    return BEGINNER_TASK_LABELS.get(
        problem_type,
        str(problem_type).lower(),
    )


def method_plain_name(method):
    """
    Honest plain name of the explanation method actually
    used. Random Forest importance is reported as feature
    importance, never as SHAP.
    """
    if method == "SHAP":
        return "SHAP"

    if method == "Model Coefficients":
        return "model coefficients (this is not SHAP)"

    if method == "Feature Importance":
        return (
            "feature importance from the model itself "
            "(this is not SHAP)"
        )

    if method == "Permutation Importance":
        return "permutation importance (this is not SHAP)"

    if method == "Cluster Profiles":
        return (
            "cluster profiles, i.e. group averages "
            "(this is not SHAP)"
        )

    return None


# ============================================================
# STRUCTURED CONTEXT BUILDING
# ============================================================

def build_explanation_context(
    context,
    max_top_features=MAX_TOP_FEATURES
):
    """
    Build the structured context that is sent to the LLM.

    This contains only computed results (metrics, feature
    importance, cluster profiles). It NEVER contains the
    raw dataset.

    The LLM explains these results; it does not compute
    anything.
    """

    payload = {
        "problem_type": context.get("problem_type"),
        "target": context.get("target"),
        "model": context.get("model"),
        "objective": context.get("objective"),
        "data_quality": context.get("data_quality"),
        "n_records": context.get("n_records"),
        "n_test": context.get("n_test"),
        "target_classes": context.get("target_classes"),
        "explanation_method": context.get("method"),
        "selection_reason": context.get("selection_reason"),
        "metrics": context.get("metrics"),
        "top_features": (
            context.get("top_features") or []
        )[:max_top_features],
        "readable_features": [
            humanize_feature_name(
                name,
                context.get("source_columns"),
            )
            for name in (
                context.get("top_features") or []
            )[:max_top_features]
        ],
        "feature_importance": (
            context.get("feature_importance") or []
        )[:max_top_features],
        "cluster_profiles": context.get(
            "cluster_profiles"
        ),
        "evaluation_notes": context.get(
            "evaluation_notes"
        )
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        default=str
    )


# ============================================================
# LLM PROMPT (required sections + honesty rules)
# ============================================================

SYSTEM_PROMPT = (
    "You are TaskAware, an explainable machine-learning "
    "assistant. You explain ML results to non-experts in "
    "plain language. "
    "You NEVER invent or compute metrics; you only explain "
    "the structured results computed by Python, using the "
    "exact numbers provided. "
    "You NEVER claim a feature causes the prediction; "
    "describe associations only (for example: records with "
    "lower viewing activity were more often in the churned "
    "group). "
    "You NEVER call Random Forest feature importance SHAP "
    "and never describe any other method as SHAP; "
    "name the explanation method exactly as given. "
    "You use the readable feature descriptions provided, "
    "never raw column names as the primary explanation. "
    "Avoid unexplained technical jargon. "
    "Every explanation must include at least one limitation."
)

USER_PROMPT_TEMPLATE = (
    "Explain the following ML analysis results for a "
    "non-expert. Use plain language.\n\n"
    "READABLE GLOSSARY (use these meanings, do not "
    "redefine them):\n"
    "- accuracy: How often the model was correct overall.\n"
    "- precision: How often a positive prediction was "
    "correct.\n"
    "- recall: How many actual positive cases the model "
    "found.\n"
    "- F1-score: A balance between precision and recall.\n"
    "- classification: Choosing between categories.\n"
    "- regression: Predicting a numerical value.\n"
    "- clustering: Finding groups of similar records.\n"
    "- feature importance: Which data fields helped the "
    "model most.\n"
    "- SHAP: How individual features contributed to a "
    "particular prediction.\n\n"
    "RULES:\n"
    "1. Explain at most five top features.\n"
    "2. State the actual explanation method used.\n"
    "3. Distinguish association from causation.\n"
    "4. Mention limitations and confidence honestly.\n"
    "5. Use each of these exact section headers, each on "
    "its own line starting with '## ':\n"
    "## What TaskAware Found\n"
    "## Why the Model Was Selected\n"
    "## What Influenced the Result\n"
    "## How Reliable Is It?\n"
    "## Important Limitation\n"
    "## Simple Conclusion\n\n"
    "STRUCTURED RESULTS (computed by Python, explain only "
    "these numbers):\n\n"
    "{structured_context}"
)


def build_user_prompt(structured_context):
    """
    Build the user prompt with required sections.
    """
    return USER_PROMPT_TEMPLATE.format(
        structured_context=structured_context
    )


# ============================================================
# LLM CALL
# ============================================================

def call_llm(
    structured_context,
    api_key=None,
    model=DEFAULT_MODEL,
    timeout=60
):
    """
    Call the Groq API with the structured context.
    """

    api_key = (
        api_key
        if api_key
        else get_api_key()
    )

    if not api_key:
        return None, (
            "No Groq API key found. Set GROQ_API_KEY "
            "to enable LLM explanations."
        )

    try:

        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(
                            structured_context
                        ),
                    },
                ],
                "temperature": 0.3,
                "max_tokens": 1200
            },
            timeout=timeout
        )

        if response.status_code != 200:

            return None, (
                f"LLM API error {response.status_code}: "
                f"{response.text[:200]}"
            )

        data = response.json()

        content = (
            data["choices"][0]["message"]["content"]
        )

        return content.strip(), None

    except Exception as e:

        return None, (
            f"LLM request failed: {str(e)}"
        )


# ============================================================
# SECTION PARSING
# ============================================================

def split_sections(text):
    """
    Split an explanation into its '## ' sections.

    Returns an ordered dict {title: body}. Text without
    section headers is returned under a single
    "What TaskAware Found" entry.
    """
    sections = {}

    current_title = None
    current_lines = []

    for line in str(text).splitlines():
        stripped = line.strip()

        if stripped.startswith("## "):
            if current_title is not None:
                sections[current_title] = (
                    "\n".join(current_lines).strip()
                )

            current_title = (
                stripped[3:].strip().rstrip(":")
            )
            current_lines = []
        else:
            current_lines.append(line)

    if current_title is not None:
        sections[current_title] = (
            "\n".join(current_lines).strip()
        )
    else:
        body = str(text).strip()

        if body:
            sections["What TaskAware Found"] = body

    return sections


# ============================================================
# FALLBACK EXPLANATION (no API key)
# ============================================================

def _fallback_plain_summary(
    problem_type,
    target,
    model,
    n_records,
    readable,
    n_clusters=None,
):
    """
    One-paragraph beginner summary in the tone of:
    what was analyzed, which model, strongest signals.
    """
    scope = (
        f"{n_records} records"
        if isinstance(n_records, int)
        else "the uploaded records"
    )

    if problem_type == "Clustering":
        if n_clusters:
            head = (
                f"TaskAware analyzed {scope} and found "
                f"{n_clusters} natural groups of similar records."
            )
        else:
            head = (
                f"TaskAware analyzed {scope} looking for "
                f"natural groups of similar records."
            )
    elif problem_type == "Regression":
        head = (
            f"TaskAware analyzed {scope} to predict a "
            f"numerical value for {target}."
        )
    else:
        head = (
            f"TaskAware analyzed {scope} to predict which "
            f"category each record belongs to for {target}."
        )

    if model and problem_type != "Clustering":
        head += f" {model} was selected as the recommended model."

    if readable:
        signals = ", ".join(readable[:3])

        if len(readable) == 1:
            head += f" The strongest signal was {signals}."
        else:
            head += f" The strongest signals were {signals}."

    return head


def fallback_explanation(context):
    """
    Produce a structured, deterministic explanation using
    templates. This guarantees the system works even when
    no LLM API key is configured.

    The fallback is honest: it only states what Python
    actually computed, names the real explanation method,
    describes associations (never causes), and always
    includes a limitation.
    """

    problem_type = context.get("problem_type")
    target = context.get("target")
    model = context.get("model")
    method = context.get("method")
    n_records = context.get("n_records")
    n_test = context.get("n_test")
    n_clusters = context.get("n_clusters")
    metrics = context.get("metrics") or {}
    selection_reason = context.get("selection_reason")
    notes = context.get("evaluation_notes")

    top_raw = list(context.get("top_features") or [])[
        :MAX_TOP_FEATURES
    ]

    readable = [
        humanize_feature_name(
            name,
            context.get("source_columns"),
        )
        for name in top_raw
    ]

    method_name = method_plain_name(method)

    sections = {}

    # ------------------------------------------------
    # 1. What TaskAware Found
    # ------------------------------------------------

    if problem_type == "Clustering":
        found = (
            f"TaskAware looked for natural groups of "
            f"similar records"
        )

        if isinstance(n_records, int):
            found += f" across {n_records} records"

        found += "."

        if n_clusters:
            found += (
                f" It found {n_clusters} clusters that best "
                f"describe the data."
            )

        profiles = context.get("cluster_profiles") or []

        if profiles and readable:
            found += (
                " The clusters differ most in "
                + ", ".join(readable[:3])
                + "."
            )

    elif problem_type == "Regression":
        found = (
            f"TaskAware analyzed the data to predict a "
            f"numerical value for {target}."
        )

        if isinstance(n_records, int):
            found = (
                f"TaskAware analyzed {n_records} records "
                f"to predict a numerical value for {target}."
            )

    else:
        found = (
            f"TaskAware analyzed the data to predict which "
            f"category each record belongs to for {target}."
        )

        if isinstance(n_records, int):
            found = (
                f"TaskAware analyzed {n_records} records "
                f"to predict which category each record "
                f"belongs to for {target}."
            )

        classes = context.get("target_classes") or []

        if len(classes) >= 2:
            found += (
                " The possible categories are "
                + ", ".join(str(c) for c in classes[:6])
                + "."
            )

    sections["What TaskAware Found"] = found

    # ------------------------------------------------
    # 2. Why the Model Was Selected
    # ------------------------------------------------

    if problem_type == "Clustering":
        if model:
            sections["Why the Model Was Selected"] = (
                f"No prediction model was needed. TaskAware "
                f"used {model}, a grouping method that "
                f"compares records and places similar ones "
                f"together."
            )
        else:
            sections["Why the Model Was Selected"] = (
                "No prediction model was needed. TaskAware "
                "used a grouping method that compares "
                "records and places similar ones together."
            )
    elif model and selection_reason:
        sections["Why the Model Was Selected"] = (
            f"{model} was selected. {selection_reason}"
        )
    elif model:
        sections["Why the Model Was Selected"] = (
            f"{model} was selected because it performed "
            f"best among the tested models on the checks "
            f"described below."
        )
    else:
        sections["Why the Model Was Selected"] = (
            "No model could be recommended from the checks "
            "performed."
        )

    # ------------------------------------------------
    # 3. What Influenced the Result
    # ------------------------------------------------

    if readable:
        influenced = (
            "The fields that helped the model most were "
            + ", ".join(readable)
            + "."
        )
    elif problem_type == "Clustering":
        influenced = (
            "TaskAware compared all usable fields to form "
            "the groups."
        )
    else:
        influenced = (
            "No individual field stood out clearly in the "
            "checks performed."
        )

    if method_name:
        influenced += (
            f" TaskAware ranked them using {method_name}."
        )

    influenced += (
        " Records with different values in these fields "
        "tended to receive different predictions. This "
        "shows an association in the data, not proof that "
        "a field causes the outcome."
    )

    sections["What Influenced the Result"] = influenced

    # ------------------------------------------------
    # 4. How Reliable Is It?
    # ------------------------------------------------

    if metrics:
        parts = []

        for key, value in metrics.items():
            gloss = BEGINNER_METRIC_LABELS.get(
                str(key).lower(), str(key)
            )
            parts.append(f"{gloss} The measured value is {value}.")

        reliable = " ".join(parts)

        if isinstance(n_test, int) and n_test > 0:
            reliable += (
                f" These checks were done on {n_test} "
                f"records the model had not seen during "
                f"training."
            )
    else:
        reliable = (
            "No performance numbers were available for "
            "this analysis, so treat the result as "
            "exploratory."
        )

    sections["How Reliable Is It?"] = reliable

    # ------------------------------------------------
    # 5. Important Limitation
    # ------------------------------------------------

    if problem_type == "Clustering":
        limitation = (
            "These clusters are patterns discovered in this "
            "dataset; they are not true or official labels. "
            "Different settings can produce different "
            "clusters."
        )
    else:
        limitation = (
            "The model learned patterns in this dataset "
            "only, and those patterns may not hold for new "
            "data."
        )

    if notes and "imbalance" in str(notes).lower():
        limitation += (
            " The categories are unevenly represented, so "
            "rare categories may be predicted less "
            "reliably."
        )

    limitation += (
        " Validate the model on real new data before "
        "operational use."
    )

    sections["Important Limitation"] = limitation

    # ------------------------------------------------
    # 6. Simple Conclusion
    # ------------------------------------------------

    if problem_type == "Clustering":
        conclusion = (
            f"The data naturally splits into "
            f"{n_clusters if n_clusters else 'several'} "
            f"clusters. Use the cluster descriptions to "
            f"decide what each cluster means for your work."
        )
    elif model and readable:
        conclusion = (
            f"{model} gives a reasonable starting point "
            f"for {target}, guided mostly by "
            + ", ".join(readable[:3])
            + ". Confirm the result on new data before "
            "relying on it."
        )
    elif model:
        conclusion = (
            f"{model} gives a reasonable starting point "
            f"for {target}. Confirm the result on new data "
            f"before relying on it."
        )
    else:
        conclusion = (
            "The analysis is exploratory. Review the "
            "details before drawing conclusions."
        )

    sections["Simple Conclusion"] = conclusion

    body = "\n\n".join(
        f"## {title}\n{sections[title]}"
        for title in SECTION_TITLES
    )

    summary = _fallback_plain_summary(
        problem_type,
        target,
        model,
        n_records,
        readable,
        n_clusters,
    )

    return f"{summary}\n\n{body}"


# ============================================================
# EXPLANATION VALIDATION
# ============================================================

_CAUSAL_PATTERNS = [
    r"\bcaus(?:e|es|ed|ing|al|ality|ation)\b",
    r"\blea[dt]s?\s+to\b",
    r"\bresult(?:s|ed|ing)?\s+in\b",
    r"\bdriv(?:e|es|ing)\b",
    r"\bprov(?:e|es|ed|en)\b",
    r"\bproof\b",
    r"\bguarantee(?:s|d)?\b",
]

_NEGATION_RE = re.compile(
    r"\b(not|n't|never|no|without|don't|doesn't|does\s+not"
    r"|do\s+not|isn't|is\s+not|aren't|are\s+not|can't"
    r"|cannot)\b"
)

_LIMITATION_CUES = [
    "limitat",
    "caveat",
    "caution",
    "be careful",
    "uncertain",
    "may not",
    "might not",
    "not guarantee",
    "validate",
    "validation",
    "before operational use",
    "before relying on it",
    "not ground-truth",
    "not ground truth",
    "not prove",
    "not proof",
    "association, not causation",
    "association not causation",
    "not causation",
    "exploratory",
]

_JARGON_TERMS = [
    "shap",
    "tf-idf",
    "tfidf",
    "one-hot",
    "one hot",
    "hyperparameter",
    "coefficient",
    "permutation",
    "embedding",
    "regulariz",
    "roc",
    "auc",
    "silhouette",
    "cross-validat",
    "estimator",
    "homogeneity",
    "davies",
    "calinski",
]

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?")


def _allowed_numbers(context):
    """
    Numbers the explanation is allowed to state: metric
    values, record counts, cluster counts and sizes, plus
    small counting numbers (1-5, e.g. 'top 3 features').
    """
    allowed = []

    metrics = context.get("metrics") or {}

    for value in metrics.values():
        try:
            allowed.append(float(value))
        except (TypeError, ValueError):
            continue

    for key in ("n_records", "n_test", "n_clusters"):
        value = context.get(key)

        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)):
            allowed.append(float(value))

    for profile in context.get("cluster_profiles") or []:
        size = profile.get("size")

        if isinstance(size, bool):
            continue

        if isinstance(size, (int, float)):
            allowed.append(float(size))

    # Numeric target categories stated in the text (e.g. 0/1).
    for cls in context.get("target_classes") or []:
        try:
            allowed.append(float(cls))
        except (TypeError, ValueError):
            continue

    return allowed


def _negated(text, start, window=40):
    """
    Whether a match at position start is negated by a
    nearby negation word before it.
    """
    before = text[max(0, start - window):start]
    return bool(_NEGATION_RE.search(before))


def validate_explanation(text, context):
    """
    Validate a generated explanation against the
    structured Python results before display.

    Checks
    ------
    - target and model names match the analysis
    - numbers come only from structured results
    - the explanation method matches the actual method
    - no unqualified causal claims
    - at least one limitation is included
    - the beginner section avoids unexplained jargon
    - at most five top features are presented

    Returns {"valid": bool, "issues": [...]}.
    """
    issues = []

    body = str(text or "")
    lowered = body.lower()

    problem_type = context.get("problem_type")
    target = context.get("target")
    model = context.get("model")
    method = context.get("method")
    metrics = context.get("metrics") or {}

    # Full list (unsliced): the check must catch a text
    # that presents more than five top features even when
    # the context carries extra names.
    top_raw = list(context.get("top_features") or [])

    # ------------------------------------------------
    # Target / model names
    # ------------------------------------------------

    if target and str(target).lower() not in lowered:
        issues.append(
            f"The target name '{target}' does not appear "
            "in the explanation."
        )

    if model and str(model).lower() not in lowered:
        issues.append(
            f"The model name '{model}' does not appear in "
            "the explanation."
        )

    if problem_type == "Clustering" and "cluster" not in lowered:
        issues.append(
            "A clustering explanation must mention clusters."
        )

    # ------------------------------------------------
    # Explanation method honesty
    # ------------------------------------------------

    affirmed = re.sub(
        r"\bnot\s+[a-z][a-z\- ]{0,30}", "", lowered
    )

    if method == "SHAP":
        if "shap" not in affirmed:
            issues.append(
                "The explanation must state that SHAP was used."
            )
    elif method in (
        "Model Coefficients",
        "Feature Importance",
        "Permutation Importance",
        "Cluster Profiles",
    ):
        keyword = {
            "Model Coefficients": "coefficient",
            "Feature Importance": "feature importance",
            "Permutation Importance": "permutation",
            "Cluster Profiles": "cluster",
        }[method]

        if keyword not in affirmed:
            issues.append(
                f"The explanation must name the actual method "
                f"({METHOD_PLAIN_NAMES[method]})."
            )

        if "shap" in affirmed:
            issues.append(
                "The explanation wrongly mentions SHAP; the "
                f"actual method was {method}."
            )
    else:
        if "shap" in affirmed:
            issues.append(
                "The explanation mentions SHAP, but no SHAP "
                "analysis was performed."
            )

    # ------------------------------------------------
    # Numbers only from structured results
    # ------------------------------------------------

    allowed = _allowed_numbers(context)

    for match in _NUMBER_RE.finditer(lowered):
        token = match.group(0)

        if "%" in token:
            try:
                value = float(token.replace("%", "")) / 100.0
            except ValueError:
                continue
        else:
            try:
                value = float(token)
            except ValueError:
                continue

        # Small counting numbers (top 3, 5 features).
        if value.is_integer() and 1 <= int(value) <= 5:
            continue

        if "." not in token and "%" not in token:
            # Whole numbers must match exactly (so 0.99 can
            # never pass as the category 1, for example).
            matched = any(value == ref for ref in allowed)
        else:
            # Decimals/percentages allow rounding slack.
            matched = any(
                abs(value - ref)
                <= max(0.004, 0.005 * abs(ref))
                for ref in allowed
            )

        if not matched:
            issues.append(
                f"The number '{token}' does not match any "
                "value computed by TaskAware."
            )
            break

    # ------------------------------------------------
    # No causal claims
    # ------------------------------------------------

    for pattern in _CAUSAL_PATTERNS:
        for match in re.finditer(pattern, lowered):
            if _negated(lowered, match.start()):
                continue

            issues.append(
                "The explanation contains a causal claim "
                f"('{match.group(0)}'). Describe associations "
                "instead."
            )
            break

        if issues and issues[-1].startswith(
            "The explanation contains a causal claim"
        ):
            break

    # ------------------------------------------------
    # At least one limitation
    # ------------------------------------------------

    if not any(cue in lowered for cue in _LIMITATION_CUES):
        issues.append(
            "The explanation must include at least one "
            "limitation."
        )

    # ------------------------------------------------
    # Beginner section avoids unexplained jargon
    # ------------------------------------------------

    sections = split_sections(body)
    first = sections.get(
        "What TaskAware Found", body[:1200]
    ).lower()

    found_jargon = sorted({
        term
        for term in _JARGON_TERMS
        if term in first
    })

    if found_jargon:
        issues.append(
            "The beginner section uses unexplained jargon: "
            + ", ".join(found_jargon)
            + "."
        )

    # ------------------------------------------------
    # At most five top features
    # ------------------------------------------------

    if top_raw:
        mentioned = set()

        for name in top_raw:
            raw = str(name).lower()
            pretty = humanize_feature_name(name).lower()

            if raw and raw in lowered:
                mentioned.add(str(name))

            if pretty and pretty in lowered:
                mentioned.add(str(name))

        if len(mentioned) > MAX_TOP_FEATURES:
            issues.append(
                "The explanation presents more than five "
                "top features."
            )

    # Unused metrics guard: metrics must exist to be
    # discussed, but invented values are caught above.
    _ = metrics

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


# ============================================================
# PRESENTATION (progressive disclosure data for the page)
# ============================================================

def build_presentation(context, explanation_result):
    """
    Build the progressive-disclosure presentation model:

    Level 1 - plain summary + section narratives.
    Level 2 - result card, top features, reliability.
    Level 3 - technical details (method, full tables).

    Pure function of the structured context and the
    validated explanation result.
    """
    metrics = context.get("metrics") or {}
    problem_type = context.get("problem_type")
    method = context.get("method")

    top_raw = list(context.get("top_features") or [])[
        :MAX_TOP_FEATURES
    ]

    importance_by_feature = {}

    for item in context.get("feature_importance") or []:
        if not isinstance(item, dict):
            continue

        key = item.get("feature")

        for value_key in ("importance", "mean_abs_shap", "value"):
            if item.get(value_key) is not None:
                importance_by_feature[str(key)] = item[value_key]
                break

    source_columns = context.get("source_columns") or {}

    top_features = []

    for name in top_raw:
        label = humanize_feature_name(name, source_columns)

        detail = None
        value = importance_by_feature.get(str(name))

        if value is not None:
            detail = f"Importance score {value}."

        source = source_columns.get(str(name))

        if source == "text":
            detail = (
                ((detail + " ") if detail else "")
                + "Derived from free-text content."
            )

        top_features.append({
            "raw": str(name),
            "label": label,
            "detail": detail,
        })

    reliability = []

    for key, value in metrics.items():
        reliability.append({
            "metric": str(key),
            "label": BEGINNER_METRIC_LABELS.get(
                str(key).lower(), str(key)
            ),
            "value": value,
        })

    sections = split_sections(
        explanation_result.get("explanation", "")
    )

    limitation = sections.get("Important Limitation", "")

    technical_importance = []

    for item in context.get("feature_importance") or []:
        if not isinstance(item, dict):
            continue

        feature = str(item.get("feature", ""))

        value = None

        for value_key in ("importance", "mean_abs_shap", "value"):
            if item.get(value_key) is not None:
                value = item[value_key]
                break

        technical_importance.append({
            "feature": feature,
            "label": humanize_feature_name(
                feature, source_columns
            ),
            "value": value,
        })

    return {
        "summary": explanation_result.get(
            "plain_summary", ""
        ) or explanation_result.get("explanation", ""),
        "sections": [
            {"title": title, "body": body}
            for title, body in sections.items()
        ],
        "result_card": {
            "task": describe_problem(problem_type),
            "problem_type": problem_type,
            "target": context.get("target"),
            "model": context.get("model"),
            "n_records": context.get("n_records"),
            "objective": context.get("objective"),
            "target_classes": context.get("target_classes"),
        },
        "top_features": top_features,
        "reliability": reliability,
        "limitation": limitation,
        "technical": {
            "method": method,
            "method_plain": (
                method_plain_name(method)
                or "No explanation method was available."
            ),
            "metrics": [
                {"metric": str(k), "value": v}
                for k, v in metrics.items()
            ],
            "feature_importance": technical_importance,
            "evaluation_notes": context.get("evaluation_notes"),
            "selection_reason": context.get("selection_reason"),
            "provenance": {
                "source": explanation_result.get("source"),
                "message": explanation_result.get("message", ""),
            },
        },
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def generate_explanation(
    context,
    use_okf=True,
    okf_manager=None,
    api_key=None,
    model=DEFAULT_MODEL
):
    """
    Generate an explanation using the OKF-aware pipeline:

        check OKF store
            -> validate reused text, reuse if honest
            -> otherwise call the LLM (or fallback)
            -> validate, repair with fallback if needed
            -> save to OKF
            -> return

    Returns
    -------
    dict
        explanation, plain_summary, sections, validation,
        source (okf/llm/fallback), okf key, message.
    """

    result = {
        "explanation": "",
        "plain_summary": "",
        "sections": [],
        "validation": {"valid": True, "issues": []},
        "source": None,
        "okf_key": None,
        "message": ""
    }

    def finalize(text, source, message=""):
        validation = validate_explanation(text, context)

        if not validation["valid"]:
            repaired = fallback_explanation(context)
            repaired_validation = validate_explanation(
                repaired, context
            )

            result["explanation"] = repaired
            result["source"] = "fallback"
            result["validation"] = repaired_validation
            result["message"] = (
                "The generated explanation failed honesty "
                "checks ("
                + "; ".join(validation["issues"][:3])
                + "); used the structured fallback instead."
            )
        else:
            result["explanation"] = text
            result["source"] = source
            result["validation"] = validation
            result["message"] = message

        try:
            first = split_sections(
                result["explanation"]
            ).get("What TaskAware Found", "")
            summary = first.split(".")[0].strip()

            if summary:
                result["plain_summary"] = summary + "."
            else:
                result["plain_summary"] = first.strip()
        except Exception:
            result["plain_summary"] = ""

        result["sections"] = [
            {"title": title, "body": body}
            for title, body in split_sections(
                result["explanation"]
            ).items()
        ]

        return result

    # ------------------------------------------------
    # 1. OKF REUSE (validated before display)
    # ------------------------------------------------

    if use_okf and okf_manager is not None:

        explanation, knowledge_key, reused = (
            okf_manager.load_knowledge(context)
        )

        if reused:

            check = validate_explanation(
                explanation, context
            )

            if check["valid"]:
                result["explanation"] = explanation
                result["source"] = "okf"
                result["okf_key"] = knowledge_key
                result["validation"] = check
                result["message"] = (
                    "Explanation reused from TaskAware's "
                    "knowledge store."
                )
                result["plain_summary"] = (
                    split_sections(explanation).get(
                        "What TaskAware Found", ""
                    ).split(".")[0].strip() + "."
                )
                result["sections"] = [
                    {"title": t, "body": b}
                    for t, b in split_sections(
                        explanation
                    ).items()
                ]

                return result

            result["message"] = (
                "A stored explanation failed current "
                "honesty checks; generating a fresh one."
            )

        result["okf_key"] = knowledge_key

    # ------------------------------------------------
    # 2. LLM CALL
    # ------------------------------------------------

    structured_context = build_explanation_context(
        context
    )

    llm_text, error = call_llm(
        structured_context,
        api_key=api_key,
        model=model
    )

    if llm_text:

        finalize(
            llm_text,
            "llm",
            "",
        )

    else:

        finalize(
            fallback_explanation(context),
            "fallback",
            (
                error
                or "Used a structured fallback explanation."
            ),
        )

    # ------------------------------------------------
    # 3. SAVE TO OKF
    # ------------------------------------------------

    if okf_manager is not None:

        saved_key = okf_manager.save_knowledge(
            context,
            result["explanation"]
        )

        if saved_key:
            result["okf_key"] = saved_key

    return result
