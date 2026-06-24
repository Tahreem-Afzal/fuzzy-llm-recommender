"""
fuzzy_engine.py
---------------
The core Fuzzy Inference System (Mamdani-style) that scores how well a
single LLM fits a user's stated priorities.

PIPELINE (for each model):
    1. FUZZIFICATION   -> crisp model specs and crisp user weights are
                          converted into fuzzy membership degrees
                          (Low / Medium / High) using triangular MFs.
    2. RULE EVALUATION -> a small IF-THEN rule base combines the
                          fuzzified model-fit per criterion with the
                          user's priority weight for that criterion.
    3. AGGREGATION     -> rule outputs are combined into a single
                          fuzzy "suitability" output set.
    4. DEFUZZIFICATION -> centroid method converts the fuzzy output
                          back into one crisp suitability score (0-100).

We build ONE shared fuzzy control system and feed it different crisp
inputs per model, rather than rebuilding the system for every model.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def _build_criterion_antecedent(name, universe, low_pt, mid_pt, high_pt):
    """
    Helper to build a 3-term (Low/Medium/High) fuzzy variable using
    triangular membership functions over a given universe of discourse.
    """
    var = ctrl.Antecedent(universe, name)
    var["low"] = fuzz.trimf(universe, [universe[0], universe[0], mid_pt])
    var["medium"] = fuzz.trimf(universe, [low_pt, mid_pt, high_pt])
    var["high"] = fuzz.trimf(universe, [mid_pt, universe[-1], universe[-1]])
    return var


class LLMFuzzyRecommender:
    """
    Builds and holds the fuzzy control systems (one per criterion) used
    to score a candidate model against user priority weights, then
    combines per-criterion scores into one overall suitability score.

    Design choice: rather than one giant FIS with 5 model-inputs + 5
    user-inputs (10 antecedents -> huge rule explosion), we run FIVE
    small, identical-shaped FISs (cost, latency, accuracy, context,
    multimodality), each combining:
        - model's fuzzified fit on that criterion
        - user's fuzzified priority on that criterion
    into a per-criterion "fit score" (0-100), then defuzzify and combine
    those 5 fit-scores with a weighted average (the weights are the
    user's own priorities, normalised). This keeps each individual FIS
    small, interpretable, and easy to grade / present, while still
    using fuzzification -> rules -> aggregation -> defuzzification for
    every criterion.
    """

    CRITERIA = ["cost", "latency", "accuracy", "context", "multimodal"]

    # universes of discourse per criterion (min, max) used to normalise
    # raw model values onto a 0-100 scale before fuzzification.
    RANGES = {
        "cost": (0.1, 5.0),        # USD per 1M tokens (lower=cheaper=better)
        "latency": (30, 150),      # tokens/sec (higher=faster=better)
        "accuracy": (60, 100),     # benchmark score (higher=better)
        "context": (8, 1000),      # context window, in K tokens (higher=better)
        "multimodal": (0.0, 1.0),  # degree of multimodal support
    }

    # whether higher raw value = better (True) or lower raw value = better (False)
    HIGHER_IS_BETTER = {
        "cost": False,
        "latency": True,
        "accuracy": True,
        "context": True,
        "multimodal": True,
    }

    def __init__(self):
        self._systems = {}
        for crit in self.CRITERIA:
            self._systems[crit] = self._build_criterion_system()

    def _build_criterion_system(self):
        """
        Build one small Mamdani FIS for a generic criterion:
            inputs : model_fit (0-100), user_priority (0-100)
            output : suitability (0-100)
        Both inputs are normalised 0-100 "goodness" scores before this
        system runs, so the SAME rule base can be reused for every
        criterion (cost, latency, accuracy, context, multimodal).
        """
        universe = np.arange(0, 101, 1)

        model_fit = _build_criterion_antecedent(
            "model_fit", universe, low_pt=25, mid_pt=50, high_pt=75
        )
        user_priority = _build_criterion_antecedent(
            "user_priority", universe, low_pt=25, mid_pt=50, high_pt=75
        )

        suitability = ctrl.Consequent(universe, "suitability")
        suitability["low"] = fuzz.trimf(universe, [0, 0, 50])
        suitability["medium"] = fuzz.trimf(universe, [20, 50, 80])
        suitability["high"] = fuzz.trimf(universe, [50, 100, 100])

        # --- Rule base ---------------------------------------------------
        # Core idea: if the user doesn't care about a criterion (low
        # priority), that criterion barely affects suitability (it stays
        # medium-ish regardless of model_fit). If the user cares a lot
        # (high priority), the model's actual fit on that criterion
        # strongly drives the suitability score.
        rules = [
            # user priority LOW -> criterion barely matters, default medium
            ctrl.Rule(user_priority["low"], suitability["medium"]),

            # user priority MEDIUM -> suitability follows model_fit, softened
            ctrl.Rule(
                user_priority["medium"] & model_fit["low"], suitability["low"]
            ),
            ctrl.Rule(
                user_priority["medium"] & model_fit["medium"],
                suitability["medium"],
            ),
            ctrl.Rule(
                user_priority["medium"] & model_fit["high"], suitability["high"]
            ),

            # user priority HIGH -> suitability follows model_fit strongly
            ctrl.Rule(
                user_priority["high"] & model_fit["low"], suitability["low"]
            ),
            ctrl.Rule(
                user_priority["high"] & model_fit["medium"], suitability["medium"]
            ),
            ctrl.Rule(
                user_priority["high"] & model_fit["high"], suitability["high"]
            ),
        ]

        system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(system)

    def _normalise(self, criterion, raw_value):
        """
        Map a raw model spec value onto a 0-100 'goodness' scale, where
        100 always means 'best possible' for that criterion (handles the
        cost criterion's inverted direction automatically).
        """
        lo, hi = self.RANGES[criterion]
        raw_value = max(lo, min(hi, raw_value))  # clip into range
        scaled = (raw_value - lo) / (hi - lo) * 100.0
        if not self.HIGHER_IS_BETTER[criterion]:
            scaled = 100.0 - scaled
        return scaled

    def score_model(self, model, user_weights, verbose=False):
        """
        Compute the overall fuzzy suitability score for one model.

        Parameters
        ----------
        model : dict
            One entry from model_database.MODEL_DB
        user_weights : dict
            e.g. {"cost": 2, "latency": 2, "accuracy": 5,
                  "context": 3, "multimodal": 5}
            Values on a 1-5 priority scale (5 = matters a lot).
        verbose : bool
            If True, return the per-criterion breakdown too.

        Returns
        -------
        overall_score : float (0-100)
        breakdown : dict (per-criterion fit/priority/suitability), only
                    populated if verbose=True
        """
        breakdown = {}
        weighted_sum = 0.0
        weight_total = 0.0

        for crit in self.CRITERIA:
            model_fit_score = self._normalise(crit, model[crit])
            user_priority_score = (user_weights.get(crit, 3) / 5.0) * 100.0

            sim = self._systems[crit]
            sim.input["model_fit"] = model_fit_score
            sim.input["user_priority"] = user_priority_score
            sim.compute()
            crit_suitability = sim.output["suitability"]

            w = user_weights.get(crit, 3)
            weighted_sum += crit_suitability * w
            weight_total += w

            if verbose:
                breakdown[crit] = {
                    "model_fit": round(model_fit_score, 1),
                    "user_priority": user_weights.get(crit, 3),
                    "suitability": round(crit_suitability, 1),
                }

        overall_score = weighted_sum / weight_total if weight_total else 0.0
        return round(overall_score, 1), breakdown

    def recommend(self, models, user_weights, required_filters=None, top_n=3, verbose=False):
        """
        Filter candidate models by hard requirements, then rank the
        remainder by fuzzy suitability score.

        required_filters : dict, optional
            e.g. {"multimodal_min": 0.5, "open_source": True,
                  "context_min": 100}
        """
        required_filters = required_filters or {}
        candidates = []

        for model in models:
            if "multimodal_min" in required_filters and \
                    model["multimodal"] < required_filters["multimodal_min"]:
                continue
            if "open_source" in required_filters and \
                    model["open_source"] != required_filters["open_source"]:
                continue
            if "context_min" in required_filters and \
                    model["context"] < required_filters["context_min"]:
                continue
            candidates.append(model)

        results = []
        for model in candidates:
            score, breakdown = self.score_model(model, user_weights, verbose=verbose)
            results.append((model["name"], score, breakdown))

        results.sort(key=lambda r: r[1], reverse=True)
        return results[:top_n]
