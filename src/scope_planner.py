from typing import Dict, List

from src.evidence_scoring import (
    candidate_allowed_for_slot,
    is_slot_selection_trustworthy,
    query_relevance_score,
    score_candidate_for_slot,
)


BUDGET_TO_SUPPORT = {
    "small": 1,
    "moderate": 2,
    "default": 3,
}


PLAN_RELIABILITY_THRESHOLDS = {
    "title": 0.82,
    "first_figure_token": 0.55,
    "summary": 0.78,
    "quant_plus_first_figure": 0.68,
    "chart_numeric": 0.72,
    "general": 0.0,
}


def _dedupe_candidates(candidates):
    seen = set()
    out = []
    for cand in candidates:
        if cand["candidate_id"] in seen:
            continue
        seen.add(cand["candidate_id"])
        out.append(cand)
    return out


def plan_select_evidence(question_text, evidence_plan, candidates, doc_context, embedding_model):
    if not candidates:
        return {
            "selected_evidence": [],
            "selected_pages": [],
            "slot_assignments": {},
            "score_breakdown": {},
            "planner_debug": {
                "fallback_reason": "no_candidates",
                "plan_reliability": 0.0,
                "slot_reliability": {},
                "candidate_reliability": {},
                "structural_feature_hits": {},
            },
            "fallback_used": True,
        }

    relevance_scores = query_relevance_score(question_text, candidates, embedding_model)
    slot_assignments = {}
    selected = []
    score_breakdown = {}
    slot_reliability = {}
    candidate_reliability = {}
    structural_feature_hits = {}
    for slot in evidence_plan.evidence_slots:
        eligible_candidates = [cand for cand in candidates if candidate_allowed_for_slot(cand, slot)]
        if not eligible_candidates:
            continue
        best = None
        best_breakdown = None
        for cand in eligible_candidates:
            breakdown = score_candidate_for_slot(
                cand,
                slot,
                question_text,
                evidence_plan.structural_constraints,
                doc_context,
                relevance_scores,
            )
            if best is None or breakdown["final"] > best_breakdown["final"]:
                best = cand
                best_breakdown = breakdown
        if best is not None:
            slot_assignments[slot] = best["candidate_id"]
            score_breakdown[best["candidate_id"]] = best_breakdown
            slot_reliability[slot] = float(best_breakdown.get("slot_reliability", 0.0))
            candidate_reliability[best["candidate_id"]] = float(best_breakdown.get("candidate_reliability", 0.0))
            structural_feature_hits[best["candidate_id"]] = dict(best_breakdown.get("structural_feature_hits", {}))
            selected.append(best)

    residual_candidates = [
        cand for cand in candidates if cand["candidate_id"] not in {c["candidate_id"] for c in selected}
    ]
    residual_scored = []
    for cand in residual_candidates:
        breakdown = score_candidate_for_slot(
            cand,
            evidence_plan.evidence_slots[0],
            question_text,
            evidence_plan.structural_constraints,
            doc_context,
            relevance_scores,
        )
        residual_scored.append((breakdown["final"], cand, breakdown))
    residual_scored.sort(key=lambda x: x[0], reverse=True)
    for _, cand, breakdown in residual_scored[: BUDGET_TO_SUPPORT.get(evidence_plan.budget, 1)]:
        if cand["candidate_id"] not in score_breakdown:
            score_breakdown[cand["candidate_id"]] = breakdown
        candidate_reliability[cand["candidate_id"]] = float(breakdown.get("candidate_reliability", 0.0))
        structural_feature_hits[cand["candidate_id"]] = dict(breakdown.get("structural_feature_hits", {}))
        selected.append(cand)

    selected = _dedupe_candidates(selected)
    selected_pages = sorted({cand.get("page") for cand in selected if isinstance(cand.get("page"), int)})
    candidate_by_id = {cand["candidate_id"]: cand for cand in candidates}
    untrusted_slots = []
    missing_slots = [slot for slot in evidence_plan.evidence_slots if slot not in slot_assignments]
    untrusted_slots.extend(missing_slots)
    for slot, cand_id in slot_assignments.items():
        cand = candidate_by_id.get(cand_id)
        breakdown = score_breakdown.get(cand_id, {})
        if cand is None or not is_slot_selection_trustworthy(cand, slot, breakdown):
            untrusted_slots.append(slot)
    unique_untrusted_slots = sorted(set(untrusted_slots))
    plan_reliability = 0.0
    if evidence_plan.evidence_slots:
        plan_reliability = sum(slot_reliability.get(slot, 0.0) for slot in evidence_plan.evidence_slots) / len(evidence_plan.evidence_slots)
    plan_reliability -= 0.2 * len(missing_slots)
    if doc_context.get("weak_parser_coverage"):
        plan_reliability -= 0.1
    fallback_reason = ""
    if missing_slots:
        fallback_reason = "missing_required_slots"
    elif unique_untrusted_slots:
        fallback_reason = "untrusted_slots"
    elif plan_reliability < PLAN_RELIABILITY_THRESHOLDS.get(evidence_plan.task_type, 0.0):
        fallback_reason = "low_plan_reliability"
    fallback_used = bool(fallback_reason)
    return {
        "selected_evidence": selected,
        "selected_pages": selected_pages,
        "slot_assignments": slot_assignments,
        "score_breakdown": score_breakdown,
        "planner_debug": {
            "candidate_count": len(candidates),
            "budget": evidence_plan.budget,
            "task_type": evidence_plan.task_type,
            "untrusted_slots": unique_untrusted_slots,
            "missing_slots": missing_slots,
            "slot_reliability": slot_reliability,
            "candidate_reliability": candidate_reliability,
            "structural_feature_hits": structural_feature_hits,
            "plan_reliability": float(plan_reliability),
            "fallback_reason": fallback_reason,
        },
        "fallback_used": fallback_used,
    }
