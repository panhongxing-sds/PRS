"""AdvW-PANDA metrics from multi-start adversarial weight perturbation runs."""

from __future__ import annotations

from panda.core.semantic_entropy import compute_ase


def advw_metrics_from_record(record: dict) -> dict:
    """
    Compute AdvW-PANDA and severity from an AdvW experiment record.

    AdvW_ASE = 1 - max_cluster_mass over adversarial generations.
    AdvW_severity = mean attack loss at PGD convergence.
    AdvW_worst = max attack loss across restarts.
    """
    runs = record.get("advw_perturb_runs") or []
    answers = [r.get("answer_normalized", "") for r in runs]
    ase = compute_ase(answers)

    losses: list[float] = []
    for r in runs:
        pcfg = r.get("perturb_config") or {}
        loss = pcfg.get("attack_loss_final")
        if loss is not None:
            losses.append(float(loss))

    severity = sum(losses) / len(losses) if losses else float("nan")
    worst = max(losses) if losses else float("nan")

    return {
        "AdvW_ASE": ase["U"],
        "AdvW_ASE_H_norm": ase["H_norm"],
        "AdvW_num_clusters": ase["num_clusters"],
        "AdvW_max_mass": ase["max_mass"],
        "AdvW_severity": severity,
        "AdvW_worst": worst,
        "advw_answers": answers,
        "n_advw_perturb": len(runs),
    }
