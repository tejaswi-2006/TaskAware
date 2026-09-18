import json
import os

import requests


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "llama-3.3-70b-versatile"

GROQ_API_URL = (
    "https://api.groq.com/openai/v1/chat/completions"
)


def get_api_key():
    """
    Read the Groq API key from environment variables.
    """
    return (
        os.environ.get("GROQ_API_KEY", "")
        or os.environ.get("GROQ_API_TOKEN", "")
    )


# ============================================================
# STRUCTURED CONTEXT BUILDING
# ============================================================

def build_explanation_context(
    context,
    max_top_features=8
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
        "metrics": context.get("metrics"),
        "top_features": (
            context.get("top_features") or []
        )[:max_top_features],
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

    system_prompt = (
        "You are TaskAware, an explainable machine-learning "
        "assistant. You explain ML results to non-experts. "
        "You NEVER invent or compute metrics. You receive "
        "structured results computed by Python and explain "
        "them in plain, honest language. Avoid unsupported "
        "claims. Mention caveats such as class imbalance or "
        "leakage when the structured data indicates them."
    )

    user_prompt = (
        "Explain the following ML analysis results in a "
        "clear, structured, human-readable way:\n\n"
        f"{structured_context}"
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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 800
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
# FALLBACK EXPLANATION (no API key)
# ============================================================

def fallback_explanation(context):
    """
    Produce a structured, deterministic explanation using
    templates. This guarantees the system works even when
    no LLM API key is configured.

    The fallback is honest: it only states what Python
    actually computed.
    """

    parts = []

    problem_type = context.get("problem_type")
    target = context.get("target")
    model = context.get("model")

    if problem_type == "Clustering":

        parts.append(
            "This is an unsupervised clustering analysis. "
            "The dataset contains no prediction target, "
            "so TaskAware discovered natural groups in the "
            "data using K-Means."
        )

        n_clusters = context.get("n_clusters")

        if n_clusters:
            parts.append(
                f"{n_clusters} clusters were identified "
                "as the configuration with the best "
                "silhouette score."
            )

        profiles = context.get("cluster_profiles")

        if profiles:

            parts.append(
                "The clusters differ most in the following "
                "features: "
                + ", ".join(
                    p.get("column", "")
                    for p in profiles[:4]
                )
                + "."
            )

        parts.append(
            "These clusters represent patterns discovered "
            "from the data and are not ground-truth labels."
        )

    else:

        parts.append(
            f"This analysis predicts the {target} column, "
            f"treated as a {problem_type.lower()} problem."
        )

        if model:
            parts.append(
                f"The selected model is {model}."
            )

        metrics = context.get("metrics")

        if metrics:
            parts.append(
                "Key performance metrics were computed by "
                "TaskAware: " + ", ".join(
                    f"{k} = {v}"
                    for k, v in metrics.items()
                ) + "."
            )

        top_features = context.get("top_features")

        if top_features:
            parts.append(
                "The most influential features are: "
                + ", ".join(top_features) + "."
            )

        notes = context.get("evaluation_notes")

        if notes:
            parts.append(notes)

        parts.append(
            "These explanations are based on the metrics "
            "and feature importance computed from the "
            "dataset, not on unverified claims."
        )

    return " ".join(parts)


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
            -> reuse if available
            -> otherwise call the LLM (or fallback)
            -> save to OKF
            -> return

    Returns
    -------
    dict
        explanation, source (okf/llm/fallback), okf key,
        message.
    """

    result = {
        "explanation": "",
        "source": None,
        "okf_key": None,
        "message": ""
    }

    # ------------------------------------------------
    # 1. OKF REUSE
    # ------------------------------------------------

    if use_okf and okf_manager is not None:

        explanation, knowledge_key, reused = (
            okf_manager.load_knowledge(context)
        )

        if reused:

            result["explanation"] = explanation
            result["source"] = "okf"
            result["okf_key"] = knowledge_key
            result["message"] = (
                "Explanation reused from TaskAware's "
                "knowledge store."
            )

            return result

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

        result["explanation"] = llm_text
        result["source"] = "llm"

    else:

        result["explanation"] = fallback_explanation(
            context
        )

        result["source"] = "fallback"

        result["message"] = (
            error
            or "Used a structured fallback explanation."
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