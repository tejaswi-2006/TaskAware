import hashlib
import json
import os


# ============================================================
# OKF KNOWLEDGE MANAGER
#
# OKF = Organised Knowledge Framework
#
# The OKF layer stores reusable explanations so that
# similar analyses do not need to call the LLM repeatedly.
#
# TaskAware does NOT let the LLM compute metrics. Python
# computes everything. The OKF stores the final explanation
# text, keyed by the structured result.
# ============================================================

DEFAULT_STORE = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    "okf_knowledge"
)


# ============================================================
# KEY GENERATION
# ============================================================

def generate_knowledge_key(context):
    """
    Build a deterministic key from the structured result.

    The key is a hash of the problem type, target, model
    and the top features. This allows the same explanation
    to be reused for identical analyses.
    """

    signature = json.dumps(
        {
            "problem_type": context.get("problem_type"),
            "target": context.get("target"),
            "model": context.get("model"),
            "top_features": context.get("top_features", []),
            "objective": (
                context.get("objective") or ""
            ).strip().lower()
        },
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(
        signature.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================
# STORE / LOAD
# ============================================================

def _path_for(
    knowledge_key,
    store_dir=None
):
    store_dir = (
        store_dir
        if store_dir
        else DEFAULT_STORE
    )

    return os.path.join(
        store_dir,
        f"{knowledge_key}.json"
    )


def save_knowledge(
    context,
    explanation,
    store_dir=None
):
    """
    Save an explanation to the OKF store.
    """

    if not explanation:
        return None

    knowledge_key = generate_knowledge_key(
        context
    )

    path = _path_for(
        knowledge_key,
        store_dir
    )

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    payload = {
        "key": knowledge_key,
        "context": {
            key: value
            for key, value in context.items()
            if not isinstance(
                value,
                (dict, list)
            )
        },
        "explanation": explanation,
        "source": "llm"
    }

    try:

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2
            )

        return knowledge_key

    except Exception:
        return None


def load_knowledge(
    context,
    store_dir=None
):
    """
    Try to reuse an existing explanation from the OKF
    store. Returns (explanation, knowledge_key, reused).
    """

    knowledge_key = generate_knowledge_key(
        context
    )

    path = _path_for(
        knowledge_key,
        store_dir
    )

    if not os.path.exists(path):

        return None, knowledge_key, False

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            payload = json.load(f)

        explanation = payload.get(
            "explanation",
            ""
        )

        if not explanation:
            return None, knowledge_key, False

        return explanation, knowledge_key, True

    except Exception:

        return None, knowledge_key, False


# ============================================================
# OKF SUMMARY
# ============================================================

def okf_summary(store_dir=None):
    """
    Summarize the current OKF knowledge store.
    """

    store_dir = (
        store_dir
        if store_dir
        else DEFAULT_STORE
    )

    if not os.path.isdir(store_dir):
        return {
            "total_entries": 0,
            "entries": []
        }

    entries = []

    for filename in os.listdir(store_dir):

        if not filename.endswith(".json"):
            continue

        path = os.path.join(
            store_dir,
            filename
        )

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                payload = json.load(f)

            context = payload.get(
                "context",
                {}
            )

            entries.append({
                "key": payload.get("key"),
                "problem_type": context.get(
                    "problem_type"
                ),
                "target": context.get("target"),
                "model": context.get("model"),
                "source": payload.get("source"),
                "file": filename
            })

        except Exception:
            continue

    entries.sort(
        key=lambda e: e.get("key", "")
    )

    return {
        "total_entries": len(entries),
        "entries": entries
    }