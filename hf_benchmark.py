"""
hf_benchmark.py
---------------
Fetches REAL accuracy/ranking data from Hugging Face's
lmarena-ai/leaderboard-dataset (the official Arena leaderboard data,
schema confirmed directly from the dataset card on huggingface.co).

This replaces the hand-estimated "accuracy" column in model_database.py
with an actual published Arena Score, normalised to a 0-100 scale.

No API key needed -- this dataset is public. Requires internet access
and the `datasets` package (pip install datasets).

Dataset columns used:
    model_name   -> string, e.g. "claude-opus-4-6"
    organization -> string, e.g. "anthropic"
    rating       -> float, Arena Score (Bradley-Terry rating, NOT 0-100)
    rank         -> int, rank within the 'overall' category
    category     -> string, e.g. "overall", "coding", "math"

We fetch the 'text' subset, 'latest' split, filtered to category
'overall', then map matching model_name values onto our local model
list (model_database.py) and normalise rating -> a 0-100 accuracy
score for use in fuzzy_engine.py.
"""

from datasets import load_dataset


# Maps our local display names (model_database.py) to the model_name
# string used in the lmarena dataset. Arena uses lowercase, hyphenated
# API-style identifiers that don't always match official marketing
# names exactly -- update this if Arena renames a model.
NAME_TO_ARENA_ID = {
    "Claude Opus 4.8": "claude-opus-4-8",
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "Claude Haiku 4.5": "claude-haiku-4-5-20251001",
    "GPT-5.4": "gpt-5.4",
    "GPT-5.4 mini": "gpt-5.4-mini-high",
    "GPT-4.1 nano": "gpt-4.1-2025-04-14",  # closest available proxy
    "Gemini 3.1 Pro": "gemini-3.1-pro-preview",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite-preview",
    "Llama 3.3 70B": "llama-3.3-70b-instruct",
    "Llama 4 Scout": "llama-4-scout",
    "Mistral Small": "mistral-small",
    "DeepSeek V3": "deepseek-v3-0324",
    "Qwen 2.5 72B": "qwen2.5-72b-instruct",
}


def fetch_arena_ratings():
    """
    Pulls the latest 'overall' category leaderboard from the real
    Hugging Face Arena dataset.

    Returns
    -------
    dict mapping arena model_name (lowercase id) -> raw rating (float)
    Returns None if the fetch fails (no internet, dataset renamed, etc).
    """
    try:
        ds = load_dataset(
            "lmarena-ai/leaderboard-dataset",
            "text",
            split="latest",
        )
    except Exception as e:
        print(f"[hf_benchmark] Could not load dataset: {e}")
        return None

    ratings = {}
    for row in ds:
        if row.get("category") == "overall":
            ratings[row["model_name"]] = row["rating"]

    if not ratings:
        print("[hf_benchmark] Dataset loaded but no 'overall' rows found "
              "-- check category naming hasn't changed.")
        return None

    return ratings


def normalise_ratings(ratings, model_names_subset=None):
    """
    Min-max normalise raw Arena ratings to a 0-100 scale.

    Parameters
    ----------
    ratings : dict {arena_id: raw_rating}
    model_names_subset : optional list of arena_ids to restrict the
        min/max calculation to (so our small candidate pool's scores
        are spread out 0-100 rather than compressed by including every
        model in the entire leaderboard in the min/max range).
    """
    if model_names_subset:
        relevant = {k: v for k, v in ratings.items() if k in model_names_subset}
    else:
        relevant = ratings

    if not relevant:
        return {}

    lo, hi = min(relevant.values()), max(relevant.values())
    span = hi - lo if hi != lo else 1.0

    return {k: round((v - lo) / span * 100, 1) for k, v in relevant.items()}


def get_real_accuracy_scores(verbose=True):
    """
    High-level function: fetch real Arena ratings and map them onto
    our local model database names, normalised 0-100.

    Returns
    -------
    dict {our_display_name: normalised_accuracy_score} for models that
    were found in the live dataset. Models not found are omitted --
    caller should fall back to the static estimate in model_database.py
    for those.
    """
    raw = fetch_arena_ratings()
    if raw is None:
        return {}

    arena_ids_we_want = list(NAME_TO_ARENA_ID.values())
    normalised = normalise_ratings(raw, model_names_subset=arena_ids_we_want)

    result = {}
    missing = []
    for display_name, arena_id in NAME_TO_ARENA_ID.items():
        if arena_id in normalised:
            result[display_name] = normalised[arena_id]
        else:
            missing.append(display_name)

    if verbose:
        print(f"[hf_benchmark] Fetched live accuracy for {len(result)}/"
              f"{len(NAME_TO_ARENA_ID)} models.")
        if missing:
            print(f"[hf_benchmark] Not found in live data (will use "
                  f"static estimate instead): {missing}")

    return result


if __name__ == "__main__":
    # Quick manual test: run `python hf_benchmark.py` to see what comes back.
    scores = get_real_accuracy_scores()
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"{name:<25} {score}/100")
