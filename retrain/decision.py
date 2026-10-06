"""
The two decisions the retraining pipeline makes, as pure functions (no torch,
no files, no network) so they can be unit-tested exhaustively.

  evaluate_trigger  -> should a retraining cycle start, and why?
  decide_deploy     -> does the candidate model replace the live one, and why?
"""

TIME_THRESHOLD_SECONDS = 24 * 3600   # retrain at least daily
CONFIDENCE_THRESHOLD = 0.70          # avg confidence below this is a "drop"
COOLDOWN_SECONDS = 10 * 60           # confidence trigger can't re-fire sooner than this
MIN_IMPROVEMENT = 0.002              # candidate must beat live by this on drifted data
MAX_CLEAN_REGRESSION = 0.01          # ...without losing more than this on clean data


def evaluate_trigger(
    *,
    now: float,
    last_retrain_time: float,
    avg_confidence: float | None,
    force: bool = False,
    time_threshold: float = TIME_THRESHOLD_SECONDS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    cooldown: float = COOLDOWN_SECONDS,
) -> tuple[bool, str]:
    """Either condition alone starts a cycle. `force` simulates a trigger for demos."""
    if force:
        return True, "forced (simulated trigger)"

    since_last = now - last_retrain_time
    if since_last >= time_threshold:
        return True, f"time threshold reached ({since_last / 3600:.1f}h since last retrain)"

    # Cooldown stops a persistently low-confidence period from retraining in a
    # loop: a fresh model needs time to show up in the logs before judging it.
    if avg_confidence is not None and avg_confidence < confidence_threshold:
        if since_last < cooldown:
            return False, f"confidence low ({avg_confidence:.3f}) but within {cooldown / 60:.0f}min cooldown"
        return True, f"confidence drop (avg {avg_confidence:.3f} < {confidence_threshold})"

    return False, "no trigger condition met"


def decide_deploy(
    *,
    current: dict,
    candidate: dict,
    min_improvement: float = MIN_IMPROVEMENT,
    max_clean_regression: float = MAX_CLEAN_REGRESSION,
) -> tuple[bool, str]:
    """
    current/candidate are {"clean": accuracy, "drifted": accuracy} measured on
    the SAME fixed held-out sets. Deploy only if the candidate is clearly better
    on drifted data (what live traffic now looks like) AND has not forgotten
    clean data (what the model already did well).
    """
    gain = candidate["drifted"] - current["drifted"]
    clean_change = candidate["clean"] - current["clean"]

    if clean_change < -max_clean_regression:
        return False, f"rejected: clean accuracy fell {-clean_change:.4f} (limit {max_clean_regression})"
    if gain < min_improvement:
        return False, f"rejected: drifted accuracy gain {gain:+.4f} below required {min_improvement}"
    return True, f"accepted: drifted accuracy {gain:+.4f}, clean accuracy {clean_change:+.4f}"
