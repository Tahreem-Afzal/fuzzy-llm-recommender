"""
main.py
-------
CLI demo for FuzzyLLM-Select (alternative to the web app in app.py).

Run:
    python main.py

Then either:
  - type a free-text use case (e.g. "I want to build a medical acne
    detection chatbot"), and the system will parse it into priority
    weights + hard filters automatically, OR
  - type 'manual' to enter priority sliders (1-5) yourself.

Type 'quit' to exit.
"""

from model_database import get_models_with_live_accuracy
from intent_parser import parse_intent_keywords
from fuzzy_engine import LLMFuzzyRecommender


CRITERIA_LABELS = {
    "cost": "Cost",
    "latency": "Speed",
    "accuracy": "Accuracy",
    "context": "Context window",
    "multimodal": "Multimodal support",
}


def print_result(rank, name, score, breakdown):
    print(f"\n{rank}. {name} — Suitability Score: {score}/100")
    if breakdown:
        for crit, vals in breakdown.items():
            label = CRITERIA_LABELS[crit]
            print(
                f"     {label:<20} fit={vals['model_fit']:>5.1f}  "
                f"user_priority={vals['user_priority']}  "
                f"-> suitability={vals['suitability']:>5.1f}"
            )


def manual_weights():
    print("\nRate how much each factor matters to you (1 = not at all, 5 = critical):")
    weights = {}
    for crit, label in CRITERIA_LABELS.items():
        while True:
            raw = input(f"  {label} [1-5, default 3]: ").strip()
            if raw == "":
                weights[crit] = 3
                break
            if raw.isdigit() and 1 <= int(raw) <= 5:
                weights[crit] = int(raw)
                break
            print("    Please enter a number between 1 and 5.")
    return weights, {}


def run():
    recommender = LLMFuzzyRecommender()
    print("Fetching model data (tries live Hugging Face benchmarks, "
          "falls back to static estimates if offline)...")
    models = get_models_with_live_accuracy()

    print("=" * 60)
    print(" FuzzyLLM-Select — Fuzzy Logic LLM Recommender")
    print("=" * 60)
    print("\nDescribe your use case (or type 'manual' for sliders, 'quit' to exit).")

    while True:
        text = input("\n> Use case: ").strip()
        if text.lower() in ("quit", "exit"):
            break

        if text.lower() == "manual":
            weights, filters = manual_weights()
        elif text == "":
            print("Please enter something, or type 'manual' / 'quit'.")
            continue
        else:
            weights, filters = parse_intent_keywords(text)
            print(f"\n[Parsed priorities] {weights}")
            if filters:
                print(f"[Hard filters applied] {filters}")

        results = recommender.recommend(
            models, weights, required_filters=filters, top_n=3, verbose=True
        )

        if not results:
            print("\nNo models matched your hard requirements. Try relaxing them.")
            continue

        print("\n--- Recommendations ---")
        for i, (name, score, breakdown) in enumerate(results, start=1):
            print_result(i, name, score, breakdown)


if __name__ == "__main__":
    run()
