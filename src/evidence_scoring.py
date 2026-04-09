import re
from typing import Dict, List

import numpy as np

from src.m3_lite import LABEL_PREFIX_RE, _has_strong_quant_cue, _normalize_ocr_quant_text, extract_target_figure_token

NUMERIC_RE = re.compile(r"\d")
NUMERIC_SPAN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
SECTION_LABEL_RE = re.compile(
    r"^(?:introduction|results|discussion|methods?|materials? and methods?|abstract|conclusion|references?)$",
    flags=re.IGNORECASE,
)
FIG_TABLE_RE = re.compile(r"\b(?:fig(?:ure)?\.?|table)\s*(?:S)?\d+[A-Za-z]?\b", flags=re.IGNORECASE)
END_MATTER_RE = re.compile(
    r"\b(references|acknowledg|bibliography|author contributions|license|creative commons|no claim to|open access|copyright)\b",
    flags=re.IGNORECASE,
)
SUMMARY_MARKER_RE = re.compile(
    r"\b(here, we|here we|this work|this study|this paper|we present|we report|we demonstrate|we herein reported|we herein report|our findings|in summary|to resolve)\b",
    flags=re.IGNORECASE,
)
QUANT_RESULT_RE = re.compile(
    r"\b(we surveyed|we measured|measured at|corresponding to|a total of|in total|we found|we observed|accuracy|yield)\b",
    flags=re.IGNORECASE,
)
QUANT_NOISE_RE = re.compile(r"\b(depicts|search terms?|timeline|criteria)\b", flags=re.IGNORECASE)
SPECULATIVE_RE = re.compile(r"\b(likely|expected|future|will change accordingly|predicted|envisioned)\b", flags=re.IGNORECASE)
COUNT_ENTITY_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s+(?:studies?|samples?|participants?|patients?|trials?|years?|months?|days?|hours?|minutes?|mW|W|MW|nm|um|ppm|K|mA)\b",
    flags=re.IGNORECASE,
)
PROMINENT_TITLE_RE = re.compile(r"^[A-Z].{20,200}$")


def _clean_text(text):
    return re.sub(r"\s+", " ", LABEL_PREFIX_RE.sub("", text or "")).strip()


def _is_noise_block_type(block_type):
    return block_type in {"footer", "page_number", "header", "page_footnote", "footnote"}


def _looks_like_author_or_affiliation(text):
    t = _clean_text(text)
    if not t:
        return False
    if "@" in t or "corresponding author" in t.lower() or "contributed equally" in t.lower():
        return True
    if re.search(r"\bdepartment of\b|\buniversity\b|\bacademy\b|\binstitute\b|\bchina\b|\bemail\b", t, flags=re.IGNORECASE):
        return True
    comma_count = t.count(",")
    digits = sum(ch.isdigit() for ch in t)
    titlecase_words = sum(1 for w in t.split() if re.match(r"^[A-Z][a-z]+", w))
    if comma_count >= 4 and digits >= 2:
        return True
    if digits >= 2 and "*" in t and titlecase_words >= 3:
        return True
    if titlecase_words >= 4 and digits >= 2 and len(t.split()) <= 20:
        return True
    return False


def _is_short_fragment(text):
    return len(_clean_text(text).split()) < 4


def _looks_fragmentary_lead(text):
    t = _clean_text(text)
    if not t:
        return True
    if t[0].islower():
        return True
    return bool(re.match(r"^(?:and|or|but|while|whereas|however|thus|therefore|because|although)\b", t, flags=re.IGNORECASE))


def _is_formula_noisy(text):
    t = text or ""
    if _has_strong_quant_cue(t):
        return False
    return "equation_inline" in t or "\\mathrm" in t or len(re.findall(r"[{}\\\\_]", t)) >= 8


def _is_end_matter_text(text):
    return bool(END_MATTER_RE.search(_clean_text(text)))


def _is_section_heading(candidate):
    text = _clean_text(candidate.get("text", ""))
    if not text:
        return False
    if SECTION_LABEL_RE.match(text):
        return True
    words = text.split()
    return len(words) <= 4 and text.upper() == text and text.isalpha()


def _is_title_like(candidate):
    text = _clean_text(candidate.get("text", ""))
    block_type = (candidate.get("block_type") or "").lower()
    word_count = len(text.split())
    return bool(text) and not _is_section_heading(candidate) and not _looks_like_author_or_affiliation(text) and (
        block_type == "title"
        or candidate["candidate_type"] == "title"
        or (candidate.get("page") == 1 and 5 <= word_count <= 24 and not text.endswith("."))
    )


def _is_numeric_statement(candidate):
    text = _clean_text(candidate.get("text", ""))
    normalized_text = _normalize_ocr_quant_text(candidate.get("text", ""))
    block_type = (candidate.get("block_type") or "").lower()
    if _is_noise_block_type(block_type) or _looks_like_author_or_affiliation(text):
        return False
    if _is_formula_noisy(text) and not _has_strong_quant_cue(normalized_text):
        return False
    if QUANT_NOISE_RE.search(text):
        return False
    if SPECULATIVE_RE.search(text):
        return False
    if FIG_TABLE_RE.search(text[:24]):
        return False
    if not NUMERIC_RE.search(normalized_text):
        return False
    if len(normalized_text.split()) < 8:
        return False
    if COUNT_ENTITY_RE.search(normalized_text):
        return True
    spans = NUMERIC_SPAN_RE.findall(normalized_text)
    if len(spans) >= 2:
        return True
    if re.search(r"\b(?:percent|fold|times|ppm)\b", normalized_text, flags=re.IGNORECASE):
        return True
    if QUANT_RESULT_RE.search(normalized_text):
        return True
    return False


def _page_number(candidate):
    page = candidate.get("page")
    return page if isinstance(page, int) else None


def _chunk_number(candidate):
    try:
        return int(candidate.get("metadata", {}).get("chunk_idx", 10**6))
    except (TypeError, ValueError):
        return 10**6


def _has_caption_link(candidate):
    if candidate.get("candidate_type") == "figure_token":
        return True
    anchors = candidate.get("anchors") or []
    if anchors:
        return True
    meta = candidate.get("metadata", {})
    return bool(meta.get("caption_hint") or meta.get("snippet") or candidate.get("has_image"))


def _looks_main_body(candidate):
    if candidate.get("metadata", {}).get("supplementary"):
        return False
    page = _page_number(candidate)
    if page is None:
        return False
    return 2 <= page <= 8


def _soft_structural_features(candidate, slot, question_text, doc_context):
    text = _clean_text(candidate.get("text", ""))
    page = _page_number(candidate)
    chunk_idx = _chunk_number(candidate)
    block_type = (candidate.get("block_type") or "").lower()
    features = {}

    if page == 1:
        features["front_page_bonus"] = 0.55
    elif page == 2:
        features["front_page_bonus"] = 0.25

    if _is_title_like(candidate):
        features["title_like_text_bonus"] = 0.75
    elif PROMINENT_TITLE_RE.match(text) and 5 <= len(text.split()) <= 24 and not text.endswith("."):
        features["title_like_text_bonus"] = 0.35

    if block_type in {"title", "paragraph", "text"} and 5 <= len(text.split()) <= 20 and chunk_idx <= 3:
        features["short_prominent_block_bonus"] = 0.3

    if slot in {"FIGURE_TOKEN", "TARGET_FIGURE_TOKEN", "QUANT_STATEMENT", "GENERAL_EVIDENCE"} and _has_caption_link(candidate):
        features["caption_linked_bonus"] = 0.35

    if slot == "TITLE" and page is not None:
        if page <= 2:
            features["section_position_bonus"] = 0.45
        elif page <= 4:
            features["section_position_bonus"] = 0.15
    elif slot == "CONTRIBUTION_SENTENCE" and page is not None:
        if page <= 2:
            features["section_position_bonus"] = 0.5
        elif page <= 4:
            features["section_position_bonus"] = 0.12
    elif slot == "QUANT_STATEMENT" and page is not None:
        if page <= 3:
            features["section_position_bonus"] = 0.35
        elif page <= 6:
            features["section_position_bonus"] = 0.18
    elif slot == "TARGET_FIGURE_TOKEN" and page is not None:
        if page <= 3:
            features["section_position_bonus"] = 0.25

    if _looks_main_body(candidate):
        features["main_body_bonus"] = 0.2

    if candidate.get("metadata", {}).get("supplementary") or re.search(r"\b(?:supplementary|appendix)\b", text, flags=re.IGNORECASE):
        features["supplementary_penalty"] = -0.75

    if _is_noise_block_type(block_type) or _is_end_matter_text(text):
        features["reference_footer_penalty"] = -1.0

    if _looks_like_author_or_affiliation(text):
        features["affiliation_penalty"] = -0.8

    if candidate.get("metadata", {}).get("virtual_page"):
        features["virtual_page_penalty"] = -0.25
    if slot == "CONTRIBUTION_SENTENCE" and _looks_fragmentary_lead(text):
        features["fragmentary_lead_penalty"] = -0.8

    return {k: float(v) for k, v in features.items() if abs(v) > 1e-9}


def _cue_agreement(candidate, slot):
    score = 0.0
    if slot == "TITLE":
        if _is_title_like(candidate) and _page_number(candidate) == 1:
            score += 0.45
        if _chunk_number(candidate) <= 2:
            score += 0.1
    elif slot == "FIGURE_TOKEN":
        if candidate.get("candidate_type") == "figure_token" and candidate.get("is_main_paper", True):
            score += 0.45
        if _page_number(candidate) in {1, 2, 3}:
            score += 0.1
    elif slot == "TARGET_FIGURE_TOKEN":
        if candidate.get("candidate_type") == "figure_token" and candidate.get("is_main_paper", True):
            score += 0.45
    elif slot == "CONTRIBUTION_SENTENCE":
        text = _clean_text(candidate.get("text", ""))
        if SUMMARY_MARKER_RE.search(text):
            score += 0.4
        if _page_number(candidate) in {1, 2}:
            score += 0.2
    elif slot == "QUANT_STATEMENT":
        text = _clean_text(candidate.get("text", ""))
        normalized_text = _normalize_ocr_quant_text(candidate.get("text", ""))
        if _is_numeric_statement(candidate):
            score += 0.3
        if QUANT_RESULT_RE.search(normalized_text):
            score += 0.35
        if COUNT_ENTITY_RE.search(normalized_text):
            score += 0.25
        if QUANT_NOISE_RE.search(text):
            score -= 0.55
        if SPECULATIVE_RE.search(text):
            score -= 0.7
    return score


def candidate_reliability_breakdown(candidate, slot, doc_context):
    text = _clean_text(candidate.get("text", ""))
    block_type = (candidate.get("block_type") or "").lower()
    page = _page_number(candidate)
    breakdown = {
        "parser_coverage_quality": 0.35 if not doc_context.get("weak_parser_coverage") else -0.25,
        "candidate_completeness": 0.0,
        "caption_linkage_quality": 0.0,
        "image_availability": 0.0,
        "page_structure_completeness": 0.0,
        "noise_region_risk": 0.0,
        "multi_cue_agreement": 0.0,
    }
    if len(text.split()) >= 6:
        breakdown["candidate_completeness"] += 0.2
    if page is not None:
        breakdown["candidate_completeness"] += 0.15
    if slot in {"FIGURE_TOKEN", "TARGET_FIGURE_TOKEN", "QUANT_STATEMENT", "GENERAL_EVIDENCE"} and _has_caption_link(candidate):
        breakdown["caption_linkage_quality"] += 0.25
    if candidate.get("has_image"):
        breakdown["image_availability"] += 0.2
    elif slot in {"FIGURE_TOKEN", "TARGET_FIGURE_TOKEN"} and doc_context.get("image_count", 0) > 0:
        breakdown["image_availability"] += 0.1
    if doc_context.get("page_count", 0) >= 2 and not candidate.get("metadata", {}).get("virtual_page"):
        breakdown["page_structure_completeness"] += 0.2
    elif doc_context.get("page_count", 0) <= 1:
        breakdown["page_structure_completeness"] -= 0.15
    if _is_noise_block_type(block_type) or _is_end_matter_text(text) or _looks_like_author_or_affiliation(text):
        breakdown["noise_region_risk"] -= 0.7
    if slot == "CONTRIBUTION_SENTENCE" and _looks_fragmentary_lead(text):
        breakdown["noise_region_risk"] -= 0.55
    if candidate.get("metadata", {}).get("supplementary"):
        breakdown["noise_region_risk"] -= 0.35
    breakdown["multi_cue_agreement"] += _cue_agreement(candidate, slot)
    breakdown["total"] = sum(v for k, v in breakdown.items() if k != "total")
    return breakdown


def compute_slot_reliability(candidate, slot, breakdown):
    score = 0.0
    score += max(0.0, breakdown.get("slot", 0.0)) * 0.28
    score += max(0.0, breakdown.get("constraint", 0.0)) * 0.18
    score += max(-0.5, breakdown.get("candidate_reliability", 0.0)) * 0.22
    score += max(-0.5, breakdown.get("reliability", 0.0)) * 0.12
    score += max(-0.5, breakdown.get("relevance", 0.0)) * 0.15
    score += max(-0.5, breakdown.get("structural_bonus", 0.0)) * 0.2
    if slot in {"FIGURE_TOKEN", "TARGET_FIGURE_TOKEN"} and candidate.get("candidate_type") == "figure_token":
        score += 0.15
    if slot == "TITLE" and _is_title_like(candidate):
        score += 0.2
    return float(score)


def candidate_allowed_for_slot(candidate, slot):
    ctype = candidate.get("candidate_type")
    block_type = (candidate.get("block_type") or "").lower()
    text = _clean_text(candidate.get("text", ""))
    if not text:
        return False
    if slot == "TITLE":
        return (
            ctype == "passage"
            and block_type in {"title", "paragraph", "text"}
            and not _is_noise_block_type(block_type)
            and not _is_short_fragment(text)
            and not _is_section_heading(candidate)
            and not _looks_like_author_or_affiliation(text)
        )
    if slot == "FIGURE_TOKEN":
        return ctype == "figure_token"
    if slot == "TARGET_FIGURE_TOKEN":
        return ctype == "figure_token"
    if slot == "CONTRIBUTION_SENTENCE":
        return (
            ctype == "passage"
            and block_type in {"paragraph", "text"}
            and not _is_noise_block_type(block_type)
            and len(text.split()) >= 8
            and not _looks_like_author_or_affiliation(text)
            and not _is_section_heading(candidate)
            and not _is_end_matter_text(text)
            and not _looks_fragmentary_lead(text)
        )
    if slot == "QUANT_STATEMENT":
        return (
            ctype == "passage"
            and block_type in {"paragraph", "text", "image"}
            and _is_numeric_statement(candidate)
            and not _looks_like_author_or_affiliation(text)
            and not _is_section_heading(candidate)
            and not _is_end_matter_text(text)
        )
    if slot == "GENERAL_EVIDENCE":
        return ctype in {"passage", "page", "image"}
    return True


SLOT_CONSTRAINTS = {
    "TITLE": {"front_page", "title_like_block"},
    "FIGURE_TOKEN": {"main_paper_only", "earliest_eligible"},
    "TARGET_FIGURE_TOKEN": {"main_paper_only", "target_figure_match"},
    "CONTRIBUTION_SENTENCE": {"abstract_or_intro", "front_matter_preferred"},
    "QUANT_STATEMENT": {"main_paper_only", "numeric_statement_required"},
    "GENERAL_EVIDENCE": set(),
}


def query_relevance_score(question_text, candidates, embedding_model):
    if not candidates:
        return {}
    candidate_texts = [_clean_text(c.get("text", "")) or c["candidate_type"] for c in candidates]
    q_emb = np.asarray(embedding_model.encode([question_text], normalize_embeddings=True, show_progress_bar=False))[0]
    cand_emb = np.asarray(embedding_model.encode(candidate_texts, normalize_embeddings=True, show_progress_bar=False))
    scores = np.dot(cand_emb, q_emb)
    return {c["candidate_id"]: float(s) for c, s in zip(candidates, scores)}


def slot_score(candidate, slot, question_text):
    block_type = (candidate.get("block_type") or "").lower()
    text = _clean_text(candidate.get("text", ""))
    if slot == "TITLE":
        score = 1.0 if _is_title_like(candidate) else 0.0
        if block_type == "title":
            score += 0.5
        if candidate.get("page") == 1:
            score += 0.3
        if int(candidate.get("metadata", {}).get("chunk_idx", 10**6)) <= 3:
            score += 0.4
        if _is_section_heading(candidate):
            score -= 2.0
        return score
    if slot == "FIGURE_TOKEN":
        score = 1.2 if candidate["candidate_type"] == "figure_token" else 0.0
        if candidate["candidate_type"] == "passage" and re.search(r"\b(?:fig(?:ure)?\.?|table)\s*(?:S)?\d+", text, flags=re.IGNORECASE):
            score += 0.6
        if candidate.get("metadata", {}).get("supplementary"):
            score -= 0.4
        return score
    if slot == "TARGET_FIGURE_TOKEN":
        score = 1.2 if candidate["candidate_type"] == "figure_token" else 0.0
        target_token = extract_target_figure_token(question_text)
        candidate_token = _clean_text(candidate.get("text", ""))
        if target_token and candidate_token:
            score += 1.1 if candidate_token.lower() == target_token.lower() else -0.8
        if candidate.get("metadata", {}).get("supplementary"):
            score -= 0.4
        return score
    if slot == "CONTRIBUTION_SENTENCE":
        score = 0.0
        if candidate["candidate_type"] == "passage":
            score += 0.5
        if SUMMARY_MARKER_RE.search(text):
            score += 1.4
        if candidate.get("page") in {1, 2}:
            score += 0.4
        if len(text.split()) >= 14:
            score += 0.3
        if _is_short_fragment(text) or _looks_like_author_or_affiliation(text) or _is_end_matter_text(text):
            score -= 1.0
        if _looks_fragmentary_lead(text):
            score -= 1.2
        return score
    if slot == "QUANT_STATEMENT":
        score = 1.0 if _is_numeric_statement(candidate) else 0.0
        normalized_text = _normalize_ocr_quant_text(candidate.get("text", ""))
        if candidate["candidate_type"] == "passage":
            score += 0.4
        if QUANT_RESULT_RE.search(normalized_text):
            score += 0.8
        if re.search(r"\b(surveyed|measured at|corresponding to|a total of|in total|power|intensit(?:y|ies))\b", normalized_text, flags=re.IGNORECASE):
            score += 0.9
        if COUNT_ENTITY_RE.search(normalized_text):
            score += 0.6
        if re.search(r"\b(?:\d[\d,]*(?:\.\d+)?\s*(?:and|,)\s*)+\d[\d,]*(?:\.\d+)?\s*(?:mW|W|MW|nm|um|%)\b", normalized_text, flags=re.IGNORECASE):
            score += 0.75
        if candidate.get("page") in {1, 2, 3}:
            score += 0.5
        elif isinstance(candidate.get("page"), int) and candidate.get("page") > 5:
            score -= 1.0
        if _chunk_number(candidate) > 80:
            score -= 0.8
        if FIG_TABLE_RE.search(text[:40]):
            score -= 0.2
        if QUANT_NOISE_RE.search(text):
            score -= 1.8
        if SPECULATIVE_RE.search(text):
            score -= 1.2
        if _looks_like_author_or_affiliation(text) or _is_formula_noisy(text) or _is_end_matter_text(text):
            score -= 1.5
        return score
    if slot == "GENERAL_EVIDENCE":
        return 0.4 if candidate["candidate_type"] in {"passage", "page"} else 0.2
    return 0.0


def constraint_score(candidate, constraint, question_text):
    page = candidate.get("page")
    block_type = (candidate.get("block_type") or "").lower()
    meta = candidate.get("metadata", {})
    text = _clean_text(candidate.get("text", ""))
    if constraint == "front_page":
        return 1.0 if page == 1 else 0.0
    if constraint == "title_like_block":
        return 1.0 if _is_title_like(candidate) else 0.0
    if constraint == "main_paper_only":
        return 1.0 if candidate.get("is_main_paper", True) else -1.0
    if constraint == "earliest_eligible":
        if not isinstance(page, int):
            return 0.0
        chunk_idx = int(meta.get("chunk_idx", 0))
        return max(0.0, 1.0 - 0.08 * max(page - 1, 0) - 0.02 * chunk_idx)
    if constraint == "abstract_or_intro":
        if not isinstance(page, int):
            return 0.0
        if page <= 2 and block_type in {"paragraph", "text"}:
            return 0.8
        return 0.0
    if constraint == "front_matter_preferred":
        return 0.8 if isinstance(page, int) and page <= 2 else 0.0
    if constraint == "numeric_statement_required":
        return 1.0 if _is_numeric_statement(candidate) else -0.5
    if constraint == "target_figure_match":
        target_token = extract_target_figure_token(question_text)
        if not target_token:
            return 0.0
        candidate_token = _clean_text(candidate.get("text", ""))
        if not candidate_token:
            return -0.5
        return 1.0 if candidate_token.lower() == target_token.lower() else -0.75
    return 0.0


def reliability_score(candidate, doc_context):
    block_type = (candidate.get("block_type") or "").lower()
    text = _clean_text(candidate.get("text", ""))
    score = 0.5
    if block_type == "title":
        score += 0.5
    if block_type in {"paragraph", "text"}:
        score += 0.2
    if _is_noise_block_type(block_type):
        score -= 0.5
    if candidate.get("candidate_type") == "figure_token":
        score += 0.2
    if candidate.get("has_image"):
        score += 0.15
    if candidate.get("metadata", {}).get("virtual_page"):
        score -= 0.2
    if doc_context.get("weak_parser_coverage"):
        score -= 0.2
    if _is_end_matter_text(text):
        score -= 0.8
    if candidate.get("metadata", {}).get("supplementary"):
        score -= 0.5
    if _looks_like_author_or_affiliation(text):
        score -= 0.8
    if _is_short_fragment(text):
        score -= 0.5
    if _looks_fragmentary_lead(text):
        score -= 0.45
    if _is_formula_noisy(text):
        score -= 0.4
    return score


def candidate_cost(candidate):
    text = _clean_text(candidate.get("text", ""))
    length_penalty = min(len(text.split()) / 80.0, 1.0) * 0.25
    redundancy_penalty = 0.15 if candidate.get("candidate_type") == "page" and len(text.split()) > 80 else 0.0
    return length_penalty + redundancy_penalty


def score_candidate_for_slot(candidate, slot, question_text, constraints, doc_context, relevance_scores):
    relevance = relevance_scores.get(candidate["candidate_id"], 0.0)
    slot_match = slot_score(candidate, slot, question_text)
    applicable_constraints = [c for c in constraints if c in SLOT_CONSTRAINTS.get(slot, set())]
    constraint_total = sum(constraint_score(candidate, constraint, question_text) for constraint in applicable_constraints)
    structural_feature_hits = _soft_structural_features(candidate, slot, question_text, doc_context)
    structural_bonus = sum(structural_feature_hits.values())
    reliability = reliability_score(candidate, doc_context)
    reliability_breakdown = candidate_reliability_breakdown(candidate, slot, doc_context)
    candidate_reliability = reliability_breakdown["total"]
    cost = candidate_cost(candidate)
    final = (
        0.45 * relevance
        + 0.9 * slot_match
        + 0.5 * constraint_total
        + 0.25 * structural_bonus
        + 0.18 * candidate_reliability
        + 0.22 * reliability
        - 0.2 * cost
    )
    slot_reliability = compute_slot_reliability(
        candidate,
        slot,
        {
            "relevance": relevance,
            "slot": slot_match,
            "constraint": constraint_total,
            "candidate_reliability": candidate_reliability,
            "reliability": reliability,
            "structural_bonus": structural_bonus,
        },
    )
    return {
        "relevance": relevance,
        "slot": slot_match,
        "constraint": constraint_total,
        "reliability": reliability,
        "structural_bonus": structural_bonus,
        "structural_feature_hits": structural_feature_hits,
        "candidate_reliability": candidate_reliability,
        "candidate_reliability_breakdown": reliability_breakdown,
        "slot_reliability": slot_reliability,
        "cost": cost,
        "final": final,
    }


def is_slot_selection_trustworthy(candidate, slot, breakdown):
    text = _clean_text(candidate.get("text", ""))
    page = candidate.get("page")
    if slot == "TITLE":
        return candidate_allowed_for_slot(candidate, slot) and breakdown.get("final", 0.0) >= 2.0 and breakdown.get("slot_reliability", 0.0) >= 0.85
    if slot == "FIGURE_TOKEN":
        return candidate.get("candidate_type") == "figure_token" and breakdown.get("final", 0.0) >= 1.2 and breakdown.get("slot_reliability", 0.0) >= 0.55
    if slot == "TARGET_FIGURE_TOKEN":
        return candidate.get("candidate_type") == "figure_token" and breakdown.get("final", 0.0) >= 1.2 and breakdown.get("slot_reliability", 0.0) >= 0.6
    if slot == "CONTRIBUTION_SENTENCE":
        return candidate_allowed_for_slot(candidate, slot) and breakdown.get("slot_reliability", 0.0) >= 0.75 and (
            SUMMARY_MARKER_RE.search(text) is not None or (isinstance(page, int) and page <= 2 and breakdown.get("final", 0.0) >= 2.0)
        )
    if slot == "QUANT_STATEMENT":
        return candidate_allowed_for_slot(candidate, slot) and breakdown.get("slot_reliability", 0.0) >= 0.6 and not QUANT_NOISE_RE.search(text) and (
            QUANT_RESULT_RE.search(text) is not None or COUNT_ENTITY_RE.search(text) is not None
        )
    return True
