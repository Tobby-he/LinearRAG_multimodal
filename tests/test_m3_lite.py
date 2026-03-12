from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m3_lite import (
    FIGURE_ROUTE,
    GENERAL_ROUTE,
    QUANT_ROUTE,
    SUMMARY_ROUTE,
    TITLE_ROUTE,
    extract_first_figure_answer,
    extract_quant_plus_figure_answer,
    extract_summary_answer,
    extract_summary_candidates,
    extract_title_answer,
    route_question,
)
from scripts.eval_m3_lite_dev import evaluate_quant_fields


def test_router_classifies_representative_phrasings():
    assert route_question("What is the exact full paper title?") == TITLE_ROUTE
    assert route_question("Combine one key quantitative statement with the first figure/table caption token.") == QUANT_ROUTE
    assert route_question("Please give the first figure/table caption token.") == FIGURE_ROUTE
    assert route_question("Summarize the main contribution in one sentence.") == SUMMARY_ROUTE
    assert route_question("What does this paper say?") == GENERAL_ROUTE


def test_title_route_avoids_reference_like_chunks():
    passages = [
        {"page": 1, "chunk_idx": 0, "block_type": "text", "text_for_embed": "ENGINEERING"},
        {"page": 1, "chunk_idx": 1, "block_type": "title", "text_for_embed": "Optical super-resolution nanothermometry via stimulated emission depletion imaging of upconverting nanoparticles"},
        {"page": 8, "chunk_idx": 99, "block_type": "text", "text_for_embed": "References"},
    ]
    answer, candidates = extract_title_answer(passages)
    assert answer == "Optical super-resolution nanothermometry via stimulated emission depletion imaging of upconverting nanoparticles"
    assert all("references" not in c["candidate"].lower() for c in candidates)


def test_title_answer_strips_parser_prefixes():
    answer, _ = extract_title_answer(
        [{"page": 1, "chunk_idx": 1, "block_type": "title", "text_for_embed": "title text Stoichiometry and architecture of the human pyruvate dehydrogenase complex"}]
    )
    assert answer == "Stoichiometry and architecture of the human pyruvate dehydrogenase complex"


def test_first_figure_returns_earliest_non_supplementary_token():
    passages = [
        {"page": 1, "chunk_idx": 0, "text_for_embed": "See Fig. S1 for supplementary setup.", "anchors": []},
        {"page": 2, "chunk_idx": 1, "text_for_embed": "Main results are shown in Fig. 1A.", "anchors": []},
        {"page": 3, "chunk_idx": 2, "text_for_embed": "Additional data appear in Table 2.", "anchors": []},
    ]
    answer, candidates = extract_first_figure_answer(passages)
    assert answer == "Fig. 1"
    assert candidates[0]["token"] == "Fig. S1"
    assert any(c["token"] == "Fig. 1" for c in candidates)


def test_output_formats_are_constrained():
    title_answer, _ = extract_title_answer(
        [{"page": 1, "chunk_idx": 1, "block_type": "title", "text_for_embed": "Stoichiometry and architecture of the human pyruvate dehydrogenase complex"}]
    )
    fig_answer, _ = extract_first_figure_answer(
        [{"page": 2, "chunk_idx": 3, "text_for_embed": "We report the main trend in Fig. 7C.", "anchors": []}]
    )
    assert title_answer == "Stoichiometry and architecture of the human pyruvate dehydrogenase complex"
    assert fig_answer == "Fig. 7"


def test_summary_candidates_prefer_front_matter_and_contribution_sentence():
    passages = [
        {"page": 1, "chunk_idx": 0, "block_type": "title", "text_for_embed": "A paper title"},
        {"page": 1, "chunk_idx": 2, "block_type": "paragraph", "text_for_embed": "Background on the field and prior work."},
        {"page": 1, "chunk_idx": 3, "block_type": "paragraph", "text_for_embed": "Here, we demonstrate programmable droplet transport with an adaptive surface."},
        {"page": 4, "chunk_idx": 0, "block_type": "paragraph", "text_for_embed": "References"},
    ]
    candidates = extract_summary_candidates(passages)
    assert candidates
    assert candidates[0]["candidate"] == "Here, we demonstrate programmable droplet transport with an adaptive surface."


def test_summary_answer_is_single_sentence_and_evidence_constrained():
    answer, candidates = extract_summary_answer(
        [
            {
                "page": 1,
                "chunk_idx": 1,
                "block_type": "paragraph",
                "text_for_embed": "This study presents a compact catalytic framework for hydroformylation. Additional details are provided later.",
            }
        ]
    )
    assert candidates
    assert answer == "This study presents a compact catalytic framework for hydroformylation."


def test_summary_route_prefers_abstract_like_sentence_over_title():
    answer, candidates = extract_summary_answer(
        [
            {"page": 1, "chunk_idx": 1, "block_type": "title", "text_for_embed": "Robust, scalable, and highly selective spirocyclic catalysts for industrial hydroformylation and isomerization-hydroformylation"},
            {
                "page": 1,
                "chunk_idx": 3,
                "block_type": "paragraph",
                "text_for_embed": "To resolve low activity and low regioselectivity in hydroformylation, we herein reported a class of spirocyclic diphosphites with industrially relevant performance.",
            },
        ]
    )
    assert candidates
    assert answer == "To resolve low activity and low regioselectivity in hydroformylation, we herein reported a class of spirocyclic diphosphites with industrially relevant performance."


def test_summary_answer_prefers_contribution_sentence_inside_abstract():
    answer, _ = extract_summary_answer(
        [
            {
                "page": 1,
                "chunk_idx": 3,
                "block_type": "paragraph",
                "text_for_embed": "Controlling droplets is crucial across many applications. Here, we introduce a bimodal actuation strategy for programmable 3D guidance of ferrofluidic droplets.",
            }
        ]
    )
    assert answer == "Here, we introduce a bimodal actuation strategy for programmable 3D guidance of ferrofluidic droplets."


def test_quant_plus_first_figure_returns_constrained_output():
    answer, quant_candidates, chosen_quant_statement, chosen_first_figure_token, figure_token_candidates = extract_quant_plus_figure_answer(
        [
            {"page": 1, "chunk_idx": 1, "block_type": "paragraph", "text_for_embed": "The method achieved 92% accuracy across 120 samples.", "anchors": []},
            {"page": 2, "chunk_idx": 3, "block_type": "paragraph", "text_for_embed": "Main setup is shown in Fig. 1A.", "anchors": []},
        ]
    )
    assert chosen_first_figure_token == "Fig. 1"
    assert chosen_quant_statement == "The method achieved 92% accuracy across 120 samples."
    assert figure_token_candidates
    assert quant_candidates
    assert answer == "Key statement: The method achieved 92% accuracy across 120 samples. First figure/table: Fig. 1"


def test_quant_field_level_evaluation():
    metrics = evaluate_quant_fields(
        "Key statement: The method achieved 92% accuracy across 120 samples. First figure/table: Fig. 1",
        {
            "token": "Fig. 1",
            "quant_statement": "The method achieved 92% accuracy across 120 samples.",
            "canonical_answer": "Key statement: The method achieved 92% accuracy across 120 samples. First figure/table: Fig. 1",
        },
    )
    assert metrics["token_match"] == 1.0
    assert metrics["quant_token_f1"] == 1.0
    assert metrics["canonical_answer_em"] == 1.0
