"""
intent_parser.py
-----------------
Converts a free-text use-case description (e.g. "I want to build a
medical acne detection chatbot") into:
    1. priority weights (1-5) for each criterion -> fed into the FIS
    2. hard requirement filters (e.g. must be multimodal) -> applied
       before fuzzy scoring even runs

This file ships TWO implementations:

  - parse_intent_keywords()  -> pure keyword/rule-based, works offline,
                                 no API key needed. Good default for
                                 your VS Code demo and for grading,
                                 since it's fully deterministic and you
                                 can explain every line.

  - parse_intent_llm()       -> calls an LLM (via the Anthropic API) to
                                 do the same mapping more flexibly.
                                 Requires ANTHROPIC_API_KEY to be set.
                                 Optional / stretch-goal use only.

Both return the exact same shape, so main.py can use either one
interchangeably.
"""

import json
import os
import re

DEFAULT_WEIGHTS = {"cost": 3, "latency": 3, "accuracy": 3, "context": 3, "multimodal": 1}

# Keyword -> (criterion, weight bump) rules. Very simple on purpose:
# this is meant to be transparent and easy to defend in a viva, not a
# state-of-the-art NLP system.
KEYWORD_RULES = [
    # --- vision / image-based tasks -> multimodal ---
    (r"\b(image|vision|photo|picture|detect|scan|x-ray|skin|acne|medical imag|"
      r"multimodal|multi-modal|video|audio)\w*",
     "multimodal", 5),

    # --- document / file processing -> context window + multimodal ---
    # PDFs/PPTX/scanned docs often contain layout, tables, or images, so
    # these tasks benefit from larger context AND some multimodal support.
    (r"\b(pdf|pptx|ppt|powerpoint|slide|slides|docx|word document|"
      r"spreadsheet|excel|scanned|ocr)\w*", "context", 4),
    (r"\b(pdf|pptx|ppt|powerpoint|slide|slides|scanned|ocr)\w*",
     "multimodal", 3),
    (r"\b(study notes|note[- ]?taking|summariz\w*|summary|extract\w* "
      r"(info|information|text|data))\b", "accuracy", 4),

    # --- recommendation / ranking systems -> accuracy matters more
    #     than raw chat speed, usually no multimodal need ---
    (r"\b(recommend\w*|ranking|ranking system|matchmak\w*)\b",
     "accuracy", 4),

    # --- conversational / interactive products -> latency matters ---
    (r"\b(chatbot|assistant|support bot|customer service|live chat|"
      r"conversational)\w*", "latency", 4),
    (r"\b(fast\w*|quick\w*|speedy|speedily|swift\w*|real[- ]?time|"
      r"low latency|responsive|instant\w*|rapid\w*)\b",
     "latency", 5),

    # --- cost sensitivity (wants it cheap) ---
    (r"\b(cheap|budget|free|low cost|affordable|inexpensive|cost[- ]?"
      r"effective)\b", "cost", 5),

    # --- accuracy / reliability critical domains ---
    (r"\b(accurate|accuracy|precise|reliable|critical|medical|legal|"
      r"diagnos|finance|financial|compliance|journal|research paper|"
      r"academic|peer[- ]?review\w*|citation\w*)\w*", "accuracy", 5),
    (r"\b(safety|safe|sensitive|health|patient|clinical)\w*",
     "accuracy", 5),

    # --- coding / dev tools ---
    (r"\b(code|coding|programming|developer|debug|software engineer)\w*",
     "accuracy", 4),

    # --- long-context use cases ---
    (r"\b(long document|book|research paper|large context|big document|"
      r"long context|transcript|whole codebase)\b", "context", 5),
]

# Phrases that explicitly say a criterion does NOT matter -- these force
# the weight down rather than up, which is a different signal than simply
# not mentioning the criterion at all (which keeps the neutral default).
COST_IRRELEVANT_PATTERN = (
    r"\b(no matter (the|what'?s the|of) cost|cost doesn'?t matter|"
    r"cost is not (a|an) (issue|concern|problem)|regardless of (the )?cost|"
    r"money (is )?no(t)? (object|issue|problem)|price (is )?no(t)? (object|issue)|"
    r"budget (is )?not (a|an) (issue|concern)|whatever the cost)\b"
)
SPEED_CRITICAL_PATTERN = (
    r"\b(work\w* fast\w*|must be fast\w*|needs? to be fast\w*|"
    r"speed (is )?(critical|essential|a priority)|no matter (the )?speed)\b"
)

# Explicit LOW-priority overrides: "minimum accuracy", "don't need much
# speed", etc. These run AFTER the main KEYWORD_RULES and explicitly SET
# (not max-merge) the weight to 1, so a later, more specific "I don't
# need much X" phrase correctly overrides an earlier general mention of
# the same topic word (e.g. "accurate technical writing... minimum
# accuracy" should end up low, not high).
LOW_PRIORITY_PATTERNS = {
    "accuracy": r"\b(minimum|minimal|low|little|low[- ]?level|basic) accuracy\b|"
                r"\baccuracy (is ?n'?t|is not|does ?n'?t matter|dont matter|"
                r"doesnt matter|not important|irrelevant)\b|"
                r"\b(don'?t|dont) need (much |high )?accuracy\b",
    "latency": r"\b(minimum|minimal|low|little) (speed|latency)\b|"
               r"\bspeed (is ?n'?t|is not|does ?n'?t matter|dont matter|"
               r"doesnt matter|not important|irrelevant)\b|"
               r"\b(don'?t|dont) need (it |to be )?fast\b|"
               r"\bspeed is not (a|an) (issue|priority)\b",
    "cost": r"\b(minimum|minimal|low|little) cost (priority|importance)\b",
    "context": r"\b(minimum|minimal|low|little|small) context\b|"
               r"\b(don'?t|dont) need (a |much )?(large |big |long )?context\b",
    "multimodal": r"\b(minimum|minimal|low|little|no) multimodal\b|"
                  r"\btext[- ]?only\b|\b(don'?t|dont) need (images?|vision|multimodal)\b|"
                  r"\b(should|shouldn'?t|must|mustn'?t|do|does|"
                  r"need to|needs to|has to|have to) "
                  r"not be multimodal\b|"
                  r"\bnot (be |need(s)? to be )?multimodal\b|"
                  r"\bno (image|vision|multimodal) (support|needed|required)\b",
}

HARD_FILTER_RULES = [
    (r"\b(image|vision|photo|picture|detect|scan|x-ray|skin|acne|"
      r"medical imag|multimodal|multi-modal)\w*", {"multimodal_min": 0.5}),
    (r"\b(pdf|pptx|ppt|powerpoint|slide|slides|scanned|ocr)\w*",
     {"multimodal_min": 0.3}),
    (r"\b(open[- ]?source|self[- ]?host|on[- ]?premise|local model)\b",
     {"open_source": True}),
    (r"\b(long document|book|research paper|large context|big document|"
      r"whole codebase)\b", {"context_min": 100}),
]


def parse_intent_keywords(text):
    """
    Deterministic keyword-based parser. No external API calls.

    Returns
    -------
    weights : dict  e.g. {"cost": 2, "latency": 4, "accuracy": 5,
                           "context": 3, "multimodal": 5}
    filters : dict  e.g. {"multimodal_min": 0.5}
    """
    text_lower = text.lower()
    weights = dict(DEFAULT_WEIGHTS)
    filters = {}

    for pattern, criterion, weight in KEYWORD_RULES:
        if re.search(pattern, text_lower):
            weights[criterion] = max(weights[criterion], weight)

    # "Cost doesn't matter" type phrases force cost weight DOWN to 1,
    # overriding any cost bump from the rules above (e.g. if the same
    # sentence also said "affordable" by mistake, explicit dismissal wins).
    if re.search(COST_IRRELEVANT_PATTERN, text_lower):
        weights["cost"] = 1

    # "Must be fast / speed is critical" forces latency weight to 5,
    # catching phrasing the main keyword list might miss (e.g. "fastly",
    # a non-standard word the regex word-boundary match can mishandle).
    if re.search(SPEED_CRITICAL_PATTERN, text_lower):
        weights["latency"] = 5

    for pattern, filt in HARD_FILTER_RULES:
        if re.search(pattern, text_lower):
            filters.update(filt)

    # Explicit "minimum X" / "X doesn't matter" overrides run LAST, so
    # they win over any earlier general keyword match on the same topic
    # (e.g. "...deep domain reasoning... with minimum accuracy" should
    # end up with accuracy LOW, even though "accuracy" alone elsewhere
    # in the sentence would normally push it high).
    for criterion, pattern in LOW_PRIORITY_PATTERNS.items():
        if re.search(pattern, text_lower):
            weights[criterion] = 1
            # If the user explicitly said they don't want multimodal,
            # also drop any multimodal_min hard filter that an earlier,
            # more general phrase in the same sentence may have set --
            # otherwise we'd weight multimodal as unimportant (1) while
            # still hard-excluding every non-multimodal model, which
            # directly contradicts what the user asked for.
            if criterion == "multimodal":
                filters.pop("multimodal_min", None)

    return weights, filters


def parse_intent_llm(text, model="claude-sonnet-4-6"):
    """
    Optional stretch-goal version: ask an LLM to produce the same
    weights/filters JSON. Requires the `anthropic` package and an
    ANTHROPIC_API_KEY environment variable.

    Falls back to parse_intent_keywords() if anything goes wrong, so
    your demo never crashes if the API key isn't set.
    """
    try:
        import anthropic
    except ImportError:
        print("[intent_parser] 'anthropic' package not installed, "
              "falling back to keyword parser.")
        return parse_intent_keywords(text)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[intent_parser] ANTHROPIC_API_KEY not set, "
              "falling back to keyword parser.")
        return parse_intent_keywords(text)

    prompt = f"""Given this use case description, output ONLY a JSON object
(no preamble, no markdown fences) with this exact shape:

{{
  "weights": {{"cost": <1-5>, "latency": <1-5>, "accuracy": <1-5>,
               "context": <1-5>, "multimodal": <1-5>}},
  "filters": {{"multimodal_min": <0.0-1.0, optional>,
               "open_source": <true/false, optional>,
               "context_min": <integer K-tokens, optional>}}
}}

5 = matters a lot, 1 = doesn't matter. Only include filter keys that
are clearly implied by the use case.

Use case: "{text}"
"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        weights = {**DEFAULT_WEIGHTS, **parsed.get("weights", {})}
        filters = parsed.get("filters", {})
        return weights, filters
    except Exception as e:
        print(f"[intent_parser] LLM parse failed ({e}), "
              "falling back to keyword parser.")
        return parse_intent_keywords(text)