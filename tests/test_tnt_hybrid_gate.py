import pandas as pd

from src.modeling.tnt_hybrid_m1_gate import gate_decision, gate_protocol
from src.modeling.tnt_hybrid_v3_m1_gate import (
    gate_decision as v3_gate_decision,
    gate_protocol as v3_gate_protocol,
)


def test_m1_gate_reuses_the_frozen_tnt_budget():
    protocol = gate_protocol()

    assert protocol.matched_unique_rows == 3330
    assert protocol.synthetic_epochs == 20
    assert protocol.gold_epochs == 20
    assert protocol.synthetic_learning_rate == 3e-4
    assert protocol.gold_learning_rate == 1e-4
    assert protocol.generated_eval_interval_epochs == 5


def test_m1_gate_requires_a_multi_metric_improvement(tmp_path, monkeypatch):
    import src.modeling.tnt_hybrid_m1_gate as gate

    monkeypatch.setattr(gate, "RESULTS_DIR", tmp_path)
    summary = pd.DataFrame(
        [
            {
                "condition": "tnt_v1_m1",
                "cer": 0.20,
                "wer": 0.60,
                "correction_f1": 0.02,
                "overcorrection_rate": 0.95,
                "correction_tp": 1,
            },
            {
                "condition": "tnt_hybrid_m1",
                "cer": 0.19,
                "wer": 0.55,
                "correction_f1": 0.08,
                "overcorrection_rate": 0.70,
                "correction_tp": 4,
            },
        ]
    )

    decision = gate_decision(summary)

    assert decision["ready"] is True
    assert decision["proceed_to_ablation"] is True
    assert all(decision["criteria"].values())
    assert (tmp_path / "m1_gate_decision.json").is_file()


def test_v3_m1_gate_keeps_the_matched_training_budget():
    protocol = v3_gate_protocol()

    assert protocol.matched_unique_rows == 3330
    assert protocol.synthetic_epochs == 20
    assert protocol.gold_epochs == 20
    assert protocol.generated_eval_interval_epochs == 5


def test_v3_gate_compares_against_v2_without_gold_test(tmp_path, monkeypatch):
    import src.modeling.tnt_hybrid_v3_m1_gate as gate

    monkeypatch.setattr(gate, "RESULTS_DIR", tmp_path)
    summary = pd.DataFrame(
        [
            {
                "condition": "tnt_hybrid_v2_m1",
                "cer": 0.20,
                "wer": 0.66,
                "correction_f1": 0.03,
                "overcorrection_rate": 0.97,
                "correction_tp": 2,
            },
            {
                "condition": "tnt_hybrid_v3_m1",
                "cer": 0.18,
                "wer": 0.64,
                "correction_f1": 0.10,
                "overcorrection_rate": 0.80,
                "correction_tp": 5,
            },
        ]
    )

    decision = v3_gate_decision(summary)

    assert decision["ready"] is True
    assert decision["v3_improves_v2_on_all_predeclared_criteria"] is True
    assert decision["gold_test_used"] is False
