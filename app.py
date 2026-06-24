"""
app.py
------
Flask web app for FuzzyLLM-Select.

Routes:
    GET  /              -> chat-style UI
    POST /api/recommend  -> takes {"message": "..."} or {"weights": {...}},
                            returns ranked recommendations as JSON

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify

from model_database import get_models, get_models_with_live_accuracy
from intent_parser import parse_intent_keywords, DEFAULT_WEIGHTS
from fuzzy_engine import LLMFuzzyRecommender

app = Flask(__name__)

recommender = LLMFuzzyRecommender()

# Fetch models once at startup. Tries live Hugging Face accuracy data
# first; falls back to static estimates automatically if offline.
# Set to get_models() instead if you want to skip the HF fetch entirely
# (e.g. for a fast local demo with no internet).
MODELS = get_models_with_live_accuracy(verbose=True)

CRITERIA_LABELS = {
    "cost": "Cost",
    "latency": "Speed",
    "accuracy": "Accuracy",
    "context": "Context window",
    "multimodal": "Multimodal support",
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True) or {}

    message = data.get("message", "").strip()
    manual_weights = data.get("weights")  # optional: {"cost": 3, ...}

    if manual_weights:
        weights = {**DEFAULT_WEIGHTS, **manual_weights}
        filters = {}
        parsed_info = None
    elif message:
        weights, filters = parse_intent_keywords(message)
        parsed_info = {"weights": weights, "filters": filters}
    else:
        return jsonify({"error": "No message or weights provided."}), 400

    results = recommender.recommend(
        MODELS, weights, required_filters=filters, top_n=3, verbose=True
    )

    if not results:
        return jsonify({
            "parsed": parsed_info,
            "results": [],
            "message": "No models matched your hard requirements. "
                       "Try relaxing them.",
        })

    formatted = []
    for name, score, breakdown in results:
        formatted.append({
            "name": name,
            "score": score,
            "breakdown": [
                {
                    "criterion": CRITERIA_LABELS[crit],
                    "model_fit": vals["model_fit"],
                    "user_priority": vals["user_priority"],
                    "suitability": vals["suitability"],
                }
                for crit, vals in breakdown.items()
            ],
        })

    return jsonify({"parsed": parsed_info, "results": formatted})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
