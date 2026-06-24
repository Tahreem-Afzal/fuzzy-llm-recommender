"""
model_database.py
------------------
Static, manually curated database of candidate LLMs with crisp specs.
Each model has values for the 5 criteria used by the fuzzy system:

    cost        -> USD per 1M tokens (input price, primary cost driver)
    latency     -> relative speed score, tokens/sec (higher = faster)
    accuracy    -> benchmark score out of 100 (composite, see notes)
    context     -> context window in thousands of tokens
    multimodal  -> 0.0 (text-only) to 1.0 (full vision support)

SOURCING (checked June 24, 2026):
- Cost: official provider pricing pages (anthropic.com/pricing,
  openai.com/api/pricing, ai.google.dev/gemini-api/docs/pricing) where
  available. Open-weight models (Llama, Mistral, DeepSeek, Qwen, Phi)
  have no single official API price since they're self-hosted or
  served by many third parties at different rates -- those costs are
  reasonable estimates based on common hosted-inference pricing
  (e.g. Together AI, Fireworks) and are labeled as such below.
- Accuracy: approximate composite of public benchmark standing
  (MMLU / Arena Elo-style ranking), NOT an official combined score --
  no single official "accuracy out of 100" metric exists across
  providers, so treat this column as a relative ranking proxy, not an
  exact published number. State this assumption in your report.
- Latency: relative ranking based on published throughput comparisons,
  not a guaranteed real-world tokens/sec figure (actual throughput
  varies by provider, region, and load).
- Context: official max context window in thousands of tokens.

IMPORTANT: LLM pricing changes every few months. Before submitting,
re-check current rates at the official pricing pages above and update
this file if needed -- that's a 10-minute task and makes your project
defensibly current rather than stale.
"""

MODEL_DB = [
    # --- Anthropic (source: anthropic.com/pricing, confirmed June 2026) ---
    {
        "name": "Claude Opus 4.8",
        "cost": 5.0,
        "latency": 55,
        "accuracy": 93,
        "context": 1000,
        "multimodal": 1.0,
        "open_source": False,
    },
    {
        "name": "Claude Sonnet 4.6",
        "cost": 3.0,
        "latency": 75,
        "accuracy": 89,
        "context": 1000,
        "multimodal": 1.0,
        "open_source": False,
    },
    {
        "name": "Claude Haiku 4.5",
        "cost": 1.0,
        "latency": 120,
        "accuracy": 80,
        "context": 200,
        "multimodal": 1.0,
        "open_source": False,
    },

    # --- OpenAI (source: openai.com/api/pricing, confirmed June 2026) ---
    {
        "name": "GPT-5.4",
        "cost": 2.5,
        "latency": 70,
        "accuracy": 90,
        "context": 270,
        "multimodal": 1.0,
        "open_source": False,
    },
    {
        "name": "GPT-5.4 mini",
        "cost": 0.75,
        "latency": 115,
        "accuracy": 80,
        "context": 270,
        "multimodal": 1.0,
        "open_source": False,
    },
    {
        "name": "GPT-4.1 nano",
        "cost": 0.10,
        "latency": 145,
        "accuracy": 68,
        "context": 128,
        "multimodal": 1.0,
        "open_source": False,
    },

    # --- Google (source: ai.google.dev/gemini-api/docs/pricing, confirmed June 2026) ---
    {
        "name": "Gemini 3.1 Pro",
        "cost": 2.0,
        "latency": 60,
        "accuracy": 90,
        "context": 1000,
        "multimodal": 1.0,
        "open_source": False,
    },
    {
        "name": "Gemini 2.5 Flash",
        "cost": 0.15,
        "latency": 130,
        "accuracy": 79,
        "context": 1000,
        "multimodal": 1.0,
        "open_source": False,
    },
    {
        "name": "Gemini 3.1 Flash-Lite",
        "cost": 0.10,
        "latency": 150,
        "accuracy": 72,
        "context": 1000,
        "multimodal": 1.0,
        "open_source": False,
    },

    # --- Open-weight models (cost = typical hosted-inference estimate,
    #     NOT an official single price; varies by host provider) ---
    {
        "name": "Llama 3.3 70B",
        "cost": 0.6,
        "latency": 90,
        "accuracy": 80,
        "context": 128,
        "multimodal": 0.0,
        "open_source": True,
    },
    {
        "name": "Llama 4 Scout",
        "cost": 0.20,
        "latency": 100,
        "accuracy": 82,
        "context": 128,
        "multimodal": 1.0,
        "open_source": True,
    },
    {
        "name": "Mistral Small",
        "cost": 0.20,
        "latency": 110,
        "accuracy": 78,
        "context": 128,
        "multimodal": 0.0,
        "open_source": True,
    },
    {
        "name": "DeepSeek V3",
        "cost": 0.27,
        "latency": 95,
        "accuracy": 85,
        "context": 64,
        "multimodal": 0.0,
        "open_source": True,
    },
    {
        "name": "Qwen 2.5 72B",
        "cost": 0.40,
        "latency": 88,
        "accuracy": 83,
        "context": 128,
        "multimodal": 0.4,
        "open_source": True,
    },
]


def get_models():
    """Return a fresh copy of the model database."""
    return [dict(m) for m in MODEL_DB]


def get_models_with_live_accuracy(verbose=True):
    """
    Same as get_models(), but tries to overwrite each model's 'accuracy'
    field with a REAL score pulled live from the Hugging Face Arena
    leaderboard dataset (see hf_benchmark.py).

    If the live fetch fails (no internet, dataset schema changed, a
    specific model isn't found), that model silently keeps its static
    estimate from MODEL_DB above -- the system always returns a
    complete, usable list either way.
    """
    models = get_models()

    try:
        from hf_benchmark import get_real_accuracy_scores
        live_scores = get_real_accuracy_scores(verbose=verbose)
    except ImportError:
        if verbose:
            print("[model_database] 'datasets' package not installed; "
                  "using static accuracy estimates. "
                  "Run: pip install datasets")
        return models
    except Exception as e:
        if verbose:
            print(f"[model_database] Live fetch failed ({e}); "
                  "using static accuracy estimates.")
        return models

    for m in models:
        if m["name"] in live_scores:
            m["accuracy"] = live_scores[m["name"]]
            m["accuracy_source"] = "live_huggingface"
        else:
            m["accuracy_source"] = "static_estimate"

    return models
