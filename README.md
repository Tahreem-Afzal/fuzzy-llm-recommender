# FuzzyLLM-Select

A fuzzy-logic-based decision support system that recommends the best-fit
LLM for a given use case, using **Mamdani fuzzy inference** (via
`scikit-fuzzy`) and **live benchmark data from Hugging Face's Arena
leaderboard dataset**.

Built as a Soft Computing / Big Data Analytics project demonstrating
fuzzy set theory applied to a real multi-criteria decision-making problem.

Two ways to use it: a web chatbot UI (`app.py`) and a CLI (`main.py`).
Both share the same fuzzy logic core.

## Quick Start

```bash
git clone https://github.com/<your-username>/fuzzyllm-select.git
cd fuzzyllm-select
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

**Requirements:** Python 3.9+ and an internet connection (for the live
Hugging Face benchmark fetch — the app still works offline, just with
static accuracy estimates instead).

## Setup

```bash
git clone <this-repo-url>
cd fuzzyllm
pip install -r requirements.txt
```

## Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Type a use case
(e.g. *"I want to build a chatbot for medical acne detection"*) or use
the manual sliders.

## Run the CLI

```bash
python main.py
```

## Project structure

| File | Role |
|---|---|
| `model_database.py` | Static dataset: crisp specs (cost, latency, context, multimodal) for ~14 current LLMs, sourced from official provider pricing pages |
| `hf_benchmark.py` | Fetches **real, live** accuracy/ranking data from `lmarena-ai/leaderboard-dataset` on Hugging Face (no API key needed). Falls back gracefully to static estimates if offline |
| `intent_parser.py` | Converts free-text use case → fuzzy priority weights (1–5) + hard filters, using keyword matching |
| `fuzzy_engine.py` | The Fuzzy Inference System: fuzzification → rule base → aggregation → defuzzification (centroid), per criterion |
| `app.py` | Flask web server + `/api/recommend` endpoint |
| `templates/index.html`, `static/style.css`, `static/script.js` | Web chatbot UI |
| `main.py` | CLI version of the same system |

## How the live Hugging Face data works

`hf_benchmark.py` calls:
```python
from datasets import load_dataset
ds = load_dataset("lmarena-ai/leaderboard-dataset", "text", split="latest")
```
This is the official Arena leaderboard dataset (public, no auth
required). It filters to `category == "overall"`, maps each model's
real Arena rating onto our local model list by name, and normalises
ratings to a 0–100 scale to feed into the fuzzy `accuracy` criterion.

**If a model in our local database isn't found under the exact name
used in the live Arena dataset** (model naming conventions drift
over time), that model automatically falls back to the static
accuracy estimate in `model_database.py` — the system always
produces a complete result either way, and it tells you (via console
log / startup message) which models used live data vs. estimates.

**Requires internet access** at startup. No API key needed — this
dataset is fully public.

## The fuzzy logic pipeline (for your report/viva)

For **each criterion** (cost, latency, accuracy, context, multimodal),
for **each candidate model**:

1. **Fuzzification** — the model's raw spec value and the user's 1–5
   priority are each normalised to 0–100, then mapped onto triangular
   membership functions for *Low / Medium / High*.
2. **Rule evaluation** — a 7-rule Mamdani rule base combines
   `model_fit` and `user_priority` fuzzy sets.
3. **Aggregation** — fuzzy outputs from fired rules combine into one
   fuzzy "suitability" output set per criterion.
4. **Defuzzification** — centroid method converts that into one crisp
   suitability score (0–100) per criterion.

The 5 per-criterion scores combine into one overall score via a
weighted average using the user's own priority weights.

## Known limitations (worth discussing in your report)

- The combination step across the 5 criteria is a plain weighted
  average, not itself fuzzy — only the per-criterion scoring uses
  fuzzy inference. This means a model that's cheap+fast+decent can
  sometimes outrank a model that's excellent on just the one
  criterion you said matters most. Worth a sensitivity-analysis
  discussion in your evaluation section.
- The keyword-based intent parser is simple and deterministic — it
  will miss phrasings it has no rule for.
- Live Arena ratings can shift week to week as new battles are
  recorded; treat results as reflecting "the leaderboard as of
  whenever you ran this," not a fixed ground truth.
- Model name matching between our local DB and the live Arena dataset
  is done by exact string match (`hf_benchmark.py`'s `NAME_TO_ARENA_ID`
  dict) — if Arena renames a model, that entry will silently fall back
  to the static estimate until the mapping is updated.

## Author

**Tahreem Afzal** — BS Artificial Intelligence, University of Management
and Technology, Lahore.

## License

MIT — see [LICENSE](LICENSE).
