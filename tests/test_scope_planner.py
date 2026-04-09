from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_candidates import collect_plan_candidates_from_context
from src.evidence_plan import build_evidence_plan
from src.evidence_scoring import constraint_score, reliability_score, score_candidate_for_slot, slot_score
from src.scope_planner import plan_select_evidence


class FakeEmbedder:
    vocab = [
        "title", "paper", "figure", "table", "summary", "contribution", "numeric", "statement",
        "main", "first", "earliest", "abstract", "introduction", "accuracy", "samples", "fig. 1"
    ]

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            t = (text or "").lower()
            rows.append([float(tok in t) for tok in self.vocab])
        return rows


def make_context():
    passage_metadata = {
        "p1": {
            "doc_id": "doc1", "page": 1, "chunk_idx": 1, "block_type": "title",
            "anchors": [], "text_for_embed": "A precise paper title for multimodal reasoning", "display_text": "A precise paper title for multimodal reasoning"
        },
        "p2": {
            "doc_id": "doc1", "page": 1, "chunk_idx": 2, "block_type": "paragraph",
            "anchors": [], "text_for_embed": "Here, we present a strong contribution sentence for the paper summary.", "display_text": "Here, we present a strong contribution sentence for the paper summary."
        },
        "p3": {
            "doc_id": "doc1", "page": 2, "chunk_idx": 1, "block_type": "paragraph",
            "anchors": ["ref:1"], "text_for_embed": "Main findings are shown in Fig. 1A and accuracy reached 92% across 120 samples.", "display_text": "Main findings are shown in Fig. 1A and accuracy reached 92% across 120 samples."
        },
        "p4": {
            "doc_id": "doc1", "page": 7, "chunk_idx": 1, "block_type": "footer",
            "anchors": [], "text_for_embed": "References", "display_text": "References"
        },
    }
    image_metadata = {
        "img1": {"doc_id": "doc1", "page": 2, "path": "images/page2_fig1.jpg", "caption_hint": "Fig. 1 main setup"}
    }
    pdf_structure_state = {
        "doc_to_page_nodes": {"doc1": ["page::doc1::1", "page::doc1::2"]},
        "page_nodes": {
            "page::doc1::1": {"page": 1, "doc_id": "doc1", "is_front_matter": True, "passage_hash_ids": ["p1", "p2"], "image_hash_ids": [], "virtual_page": False},
            "page::doc1::2": {"page": 2, "doc_id": "doc1", "is_front_matter": True, "passage_hash_ids": ["p3"], "image_hash_ids": ["img1"], "virtual_page": False},
        },
        "visual_nodes": {},
    }
    return passage_metadata, image_metadata, pdf_structure_state


def test_evidence_plan_creation_for_all_routes():
    for route in ["title", "first_figure_token", "summary", "quant_plus_first_figure", "general"]:
        plan = build_evidence_plan("q", route, {"predicted_route": route}, target_doc_id="doc1")
        assert plan.task_type == route
        assert plan.evidence_slots


def test_candidate_collection_returns_passage_page_figure_and_image_candidates():
    passage_metadata, image_metadata, pdf_structure_state = make_context()
    plan = build_evidence_plan("what is the title", "title", {}, target_doc_id="doc1")
    candidates, doc_context = collect_plan_candidates_from_context(passage_metadata, image_metadata, pdf_structure_state, "doc1", plan, {})
    candidate_types = {c["candidate_type"] for c in candidates}
    assert "passage" in candidate_types
    assert "page" in candidate_types
    assert "figure_token" in candidate_types
    assert "image" in candidate_types
    assert doc_context["page_count"] == 2


def test_slot_and_constraint_scoring_behavior():
    title_candidate = {"candidate_type": "passage", "block_type": "title", "page": 1, "text": "A precise paper title", "metadata": {}, "is_main_paper": True, "has_image": False}
    figure_candidate = {"candidate_type": "figure_token", "block_type": "figure_token", "page": 2, "text": "Fig. 1", "metadata": {"supplementary": False, "chunk_idx": 1}, "is_main_paper": True, "has_image": False}
    assert slot_score(title_candidate, "TITLE", "what is the title") > 1.0
    assert constraint_score(title_candidate, "front_page", "q") == 1.0
    assert slot_score(figure_candidate, "FIGURE_TOKEN", "which figure") > 1.0
    assert constraint_score(figure_candidate, "earliest_eligible", "q") > 0.0


def test_reliability_scoring_penalizes_weak_coverage_and_noise():
    noisy = {"candidate_type": "passage", "block_type": "footer", "page": 7, "text": "References", "metadata": {}, "has_image": False}
    clean = {"candidate_type": "passage", "block_type": "title", "page": 1, "text": "A good title", "metadata": {}, "has_image": False}
    weak_doc = {"weak_parser_coverage": True}
    strong_doc = {"weak_parser_coverage": False}
    assert reliability_score(clean, strong_doc) > reliability_score(noisy, weak_doc)


def test_soft_structural_features_replace_rigid_title_checks():
    candidate = {
        "candidate_id": "passage::p1",
        "candidate_type": "passage",
        "doc_id": "doc1",
        "page": 2,
        "text": "Multimodal evidence planning for robust document question answering",
        "block_type": "paragraph",
        "anchors": [],
        "is_main_paper": True,
        "has_image": False,
        "metadata": {"chunk_idx": 1},
        "source_node_ids": ["p1"],
    }
    breakdown = score_candidate_for_slot(
        candidate,
        "TITLE",
        "give the full article title",
        ["front_page", "title_like_block"],
        {"page_count": 4, "image_count": 1, "weak_parser_coverage": False},
        {"passage::p1": 0.4},
    )
    assert breakdown["structural_feature_hits"]["title_like_text_bonus"] > 0.0
    assert breakdown["structural_feature_hits"]["short_prominent_block_bonus"] > 0.0
    assert breakdown["slot_reliability"] > 0.0


def test_candidate_reliability_penalizes_virtual_page_weak_coverage():
    candidate = {
        "candidate_id": "passage::p2",
        "candidate_type": "passage",
        "doc_id": "doc1",
        "page": 1,
        "text": "References",
        "block_type": "footer",
        "anchors": [],
        "is_main_paper": True,
        "has_image": False,
        "metadata": {"chunk_idx": 9, "virtual_page": True},
        "source_node_ids": ["p2"],
    }
    breakdown = score_candidate_for_slot(
        candidate,
        "CONTRIBUTION_SENTENCE",
        "what does this paper mainly contribute",
        ["abstract_or_intro", "front_matter_preferred"],
        {"page_count": 1, "image_count": 0, "weak_parser_coverage": True},
        {"passage::p2": 0.1},
    )
    assert breakdown["candidate_reliability"] < 0.0
    assert breakdown["candidate_reliability_breakdown"]["noise_region_risk"] < 0.0


def test_planner_selects_title_candidate():
    passage_metadata, image_metadata, pdf_structure_state = make_context()
    plan = build_evidence_plan("what is the title of this paper", "title", {}, target_doc_id="doc1")
    candidates, doc_context = collect_plan_candidates_from_context(passage_metadata, image_metadata, pdf_structure_state, "doc1", plan, {})
    result = plan_select_evidence("what is the title of this paper", plan, candidates, doc_context, FakeEmbedder())
    assert result["slot_assignments"]["TITLE"].startswith("passage::p1") or result["slot_assignments"]["TITLE"].startswith("page::")


def test_planner_selects_first_figure_and_quant_slots():
    passage_metadata, image_metadata, pdf_structure_state = make_context()
    plan = build_evidence_plan("provide one key numeric finding together with the first main-paper figure/table token", "quant_plus_first_figure", {}, target_doc_id="doc1")
    candidates, doc_context = collect_plan_candidates_from_context(passage_metadata, image_metadata, pdf_structure_state, "doc1", plan, {})
    result = plan_select_evidence("provide one key numeric finding together with the first main-paper figure/table token", plan, candidates, doc_context, FakeEmbedder())
    assert "FIGURE_TOKEN" in result["slot_assignments"]
    assert "QUANT_STATEMENT" in result["slot_assignments"]


def test_planner_falls_back_cleanly_when_no_candidates():
    plan = build_evidence_plan("q", "summary", {}, target_doc_id="doc1")
    result = plan_select_evidence("q", plan, [], {"weak_parser_coverage": True}, FakeEmbedder())
    assert result["fallback_used"] is True
    assert result["selected_evidence"] == []
    assert result["planner_debug"]["fallback_reason"] == "no_candidates"


def test_planner_flags_untrusted_quant_slot_for_fallback():
    candidates = [
        {
            "candidate_id": "figure_token::doc1::2::1::Fig. 1",
            "candidate_type": "figure_token",
            "doc_id": "doc1",
            "page": 2,
            "text": "Fig. 1",
            "block_type": "figure_token",
            "anchors": [],
            "is_main_paper": True,
            "has_image": False,
            "metadata": {"chunk_idx": 1, "supplementary": False},
            "source_node_ids": [],
        },
        {
            "candidate_id": "passage::bad_quant",
            "candidate_type": "passage",
            "doc_id": "doc1",
            "page": 9,
            "text": "Timeline depicts the evolution of biomaterials in the field.",
            "block_type": "image",
            "anchors": [],
            "is_main_paper": True,
            "has_image": True,
            "metadata": {"chunk_idx": 2},
            "source_node_ids": [],
        },
    ]
    plan = build_evidence_plan(
        "provide one key numeric finding together with the first main-paper figure/table token",
        "quant_plus_first_figure",
        {},
        target_doc_id="doc1",
    )
    result = plan_select_evidence(
        "provide one key numeric finding together with the first main-paper figure/table token",
        plan,
        candidates,
        {"weak_parser_coverage": False},
        FakeEmbedder(),
    )
    assert result["fallback_used"] is True
    assert "QUANT_STATEMENT" in result["planner_debug"].get("untrusted_slots", [])
    assert result["planner_debug"]["fallback_reason"] in {"missing_required_slots", "untrusted_slots"}
    assert result["planner_debug"]["plan_reliability"] < 0.68


def test_planner_exposes_structural_and_reliability_debug_fields():
    passage_metadata, image_metadata, pdf_structure_state = make_context()
    plan = build_evidence_plan("what is the title of this paper", "title", {}, target_doc_id="doc1")
    candidates, doc_context = collect_plan_candidates_from_context(passage_metadata, image_metadata, pdf_structure_state, "doc1", plan, {})
    result = plan_select_evidence("what is the title of this paper", plan, candidates, doc_context, FakeEmbedder())
    title_id = result["slot_assignments"]["TITLE"]
    assert isinstance(result["planner_debug"]["structural_feature_hits"].get(title_id, {}), dict)
    assert title_id in result["planner_debug"]["candidate_reliability"]
    assert "TITLE" in result["planner_debug"]["slot_reliability"]
    assert isinstance(result["planner_debug"]["plan_reliability"], float)
