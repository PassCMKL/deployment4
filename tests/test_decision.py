from decision import decide_deploy, evaluate_trigger

NOW = 1_000_000.0


def test_no_trigger_when_recent_and_confident():
    fired, _ = evaluate_trigger(now=NOW, last_retrain_time=NOW - 60, avg_confidence=0.95)
    assert not fired


def test_time_threshold_triggers():
    fired, why = evaluate_trigger(now=NOW, last_retrain_time=NOW - 25 * 3600, avg_confidence=0.95)
    assert fired and "time threshold" in why


def test_confidence_drop_triggers_after_cooldown():
    fired, why = evaluate_trigger(now=NOW, last_retrain_time=NOW - 3600, avg_confidence=0.5)
    assert fired and "confidence" in why


def test_confidence_drop_blocked_by_cooldown():
    fired, why = evaluate_trigger(now=NOW, last_retrain_time=NOW - 60, avg_confidence=0.5)
    assert not fired and "cooldown" in why


def test_missing_confidence_does_not_trigger():
    assert not evaluate_trigger(now=NOW, last_retrain_time=NOW - 60, avg_confidence=None)[0]


def test_force_always_triggers():
    assert evaluate_trigger(now=NOW, last_retrain_time=NOW, avg_confidence=1.0, force=True)[0]


def test_deploys_when_clearly_better():
    ok, _ = decide_deploy(current={"clean": 0.98, "drifted": 0.70}, candidate={"clean": 0.98, "drifted": 0.85})
    assert ok


def test_rejects_when_not_better():
    ok, _ = decide_deploy(current={"clean": 0.98, "drifted": 0.70}, candidate={"clean": 0.98, "drifted": 0.70})
    assert not ok


def test_rejects_catastrophic_forgetting_even_if_drifted_improves():
    ok, why = decide_deploy(current={"clean": 0.98, "drifted": 0.70}, candidate={"clean": 0.90, "drifted": 0.95})
    assert not ok and "clean" in why
