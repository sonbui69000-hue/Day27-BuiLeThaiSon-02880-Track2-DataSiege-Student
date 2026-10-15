"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


def _valid(result):
    return isinstance(result, dict) and "error" not in result


def _outside(value, low=None, high=None):
    return (low is not None and value < low) or (high is not None and value > high)


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    profile = ctx.tools.batch_profile(payload["batch_id"])
    if not _valid(profile):
        return Verdict(alert=False, pillar="checks", reason="profile unavailable")

    b = ctx.baseline
    null_rate = profile.get("null_rate", {}).get("customer_id")
    checks = {
        "volume anomaly": _outside(
            profile.get("row_count"), b.get("row_count_min"), b.get("row_count_max")
        ),
        "null spike": null_rate is not None and null_rate > b.get("null_rate_max", float("inf")),
        "amount distribution shift": _outside(
            profile.get("mean_amount"), b.get("mean_amount_min"), b.get("mean_amount_max")
        ),
        "freshness lag": profile.get("staleness_min", 0) > b.get("staleness_min_max", float("inf")),
    }
    direct = [name for name, flagged in checks.items() if flagged]
    if direct:
        return Verdict(alert=True, pillar="checks", reason=direct[0])

    row_center = (b["row_count_min"] + b["row_count_max"]) / 2
    row_scale = (b["row_count_max"] - b["row_count_min"]) / 2
    amount_center = (b["mean_amount_min"] + b["mean_amount_max"]) / 2
    amount_scale = (b["mean_amount_max"] - b["mean_amount_min"]) / 2
    components = [
        (profile.get("row_count", row_center) - row_center) / row_scale,
        (profile.get("mean_amount", amount_center) - amount_center) / amount_scale,
        (null_rate or 0) / b["null_rate_max"],
        profile.get("staleness_min", 0) / b["staleness_min_max"],
    ]
    norm2 = sum(value * value for value in components) ** 0.5
    near_edge = sum(abs(value) > 0.75 for value in components)
    if norm2 > 1.30 and near_edge >= 3:
        return Verdict(alert=True, pillar="checks", reason="multi signal drift")
    return Verdict(alert=False, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(
        payload["contract_id"], payload["checkpoint_batch_id"]
    )
    if not _valid(diff):
        return Verdict(alert=False, pillar="contracts", reason="contract diff unavailable")

    violations = diff.get("violations", [])
    if violations:
        return Verdict(alert=True, pillar="contracts", reason=violations[0])

    declared_sla = payload.get("declared_sla", {})
    sla_limit = declared_sla.get("freshness_min")
    baseline_limit = ctx.baseline.get("freshness_delay_max_min")
    delay = diff.get("freshness_delay_min")
    if delay is not None and (
        (sla_limit is not None and delay > sla_limit)
        or (baseline_limit is not None and delay > baseline_limit)
    ):
        return Verdict(alert=True, pillar="contracts", reason="freshness SLA breach")
    return Verdict(alert=False, pillar="contracts")


def check_lineage_run(payload, ctx):
    lineage = ctx.tools.lineage_graph_slice(payload["run_id"])
    if not _valid(lineage):
        return Verdict(alert=False, pillar="lineage", reason="lineage slice unavailable")

    if lineage.get("actual_downstream_count") == 0:
        return Verdict(alert=True, pillar="lineage", reason="orphan output")

    upstream = set(lineage.get("actual_upstream", []))
    missing = {"raw.orders", "raw.customers"} - upstream
    if missing:
        return Verdict(alert=True, pillar="lineage", reason="missing upstream")

    limit = ctx.baseline.get("lineage_duration_ms_max")
    if limit is not None and lineage.get("duration_ms", 0) > limit:
        return Verdict(alert=True, pillar="lineage", reason="runtime anomaly")
    return Verdict(alert=False, pillar="lineage")


def check_feature_materialization(payload, ctx):
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if not _valid(drift):
        return Verdict(alert=False, pillar="ai_infra", reason="feature drift unavailable")

    shift = drift.get("mean_shift_sigma")
    if shift is not None and shift > 0.65:
        return Verdict(alert=True, pillar="ai_infra", reason="feature mean shift")
    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if not _valid(drift):
        return Verdict(alert=False, pillar="ai_infra", reason="embedding drift unavailable")

    if drift.get("centroid_shift", 0) > 0.04:
        return Verdict(alert=True, pillar="ai_infra", reason="embedding centroid drift")
    if drift.get("avg_doc_age_days", 0) >= 48:
        return Verdict(alert=True, pillar="ai_infra", reason="stale corpus")
    return Verdict(alert=False, pillar="ai_infra")
