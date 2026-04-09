from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m3_lite import (
    CHART_NUMERIC_ROUTE,
    FIGURE_ROUTE,
    GENERAL_ROUTE,
    QUANT_ROUTE,
    SUMMARY_ROUTE,
    TITLE_ROUTE,
    analyze_question_structure,
    _normalize_chart_numeric_text,
    extract_best_quant_statement,
    extract_chart_numeric_answer,
    extract_first_figure_answer,
    extract_presence_check_answer,
    extract_quant_plus_figure_answer,
    extract_summary_answer,
    extract_summary_candidates,
    extract_target_figure_token,
    extract_title_answer,
    route_question,
)
from scripts.eval_m3_lite_dev import evaluate_quant_fields


def test_router_classifies_representative_phrasings():
    assert route_question("What is the exact full paper title?") == TITLE_ROUTE
    assert route_question("Please give the complete paper title.") == TITLE_ROUTE
    assert route_question("According to Fig. 2, how many studies were initially identified?") == CHART_NUMERIC_ROUTE
    assert route_question("In document id sciadv.ado6268.PMC466949, according to Fig. 1A, what working distance does the dry air objective lens use?") == CHART_NUMERIC_ROUTE
    assert route_question("In document id sciadv.ado6268.PMC466949, according to Fig. 5C, what total scan time was used for each ratio map?") == CHART_NUMERIC_ROUTE
    assert route_question("In document id sciadv.abq0997.PMC466960, according to the Fig. 7B table, what maximum duration in days is listed for synthetic polymer (excluding silicone)?") == CHART_NUMERIC_ROUTE
    assert route_question("Combine one key quantitative statement with the first figure/table caption token.") == QUANT_ROUTE
    assert route_question("Report one quantitative finding from the study and also name the earliest figure/table label in the main article.") == QUANT_ROUTE
    assert route_question("Please give the first figure/table caption token.") == FIGURE_ROUTE
    assert route_question("Summarize the main contribution in one sentence.") == SUMMARY_ROUTE
    assert route_question("What does this paper say?") == GENERAL_ROUTE


def test_question_structure_prefers_structural_chart_lookup_over_keyword_coverage():
    profile = analyze_question_structure(
        "In document id sciadv.abq0997.PMC466960, according to the Fig. 7B table, what average duration in days is listed for combination biomaterials?"
    )
    assert profile["has_target_reference"] is True
    assert profile["refers_to_table"] is True
    assert profile["seeks_value"] is True
    assert profile["is_chart_numeric"] is True


def test_question_structure_avoids_open_ended_figure_questions():
    profile = analyze_question_structure("According to Fig. 3, what does this figure show about biomaterials?")
    assert profile["has_target_reference"] is True
    assert profile["asks_open_ended_figure"] is True
    assert profile["is_chart_numeric"] is False


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


def test_quant_extraction_prefers_real_numeric_finding_over_timeline_noise():
    answer, quant_candidates, chosen_quant_statement, chosen_first_figure_token, _ = extract_quant_plus_figure_answer(
        [
            {"page": 6, "chunk_idx": 2, "block_type": "paragraph", "text_for_embed": "This timeline depicts the advancement of biomaterials from the 1950s to 2050.", "anchors": []},
            {"page": 2, "chunk_idx": 1, "block_type": "paragraph", "text_for_embed": "We surveyed 834 studies in the ClinicalTrials.gov database and explored biomaterial types, their initiation points, and durations in clinical trials.", "anchors": []},
            {"page": 1, "chunk_idx": 1, "block_type": "paragraph", "text_for_embed": "Main setup is shown in Fig. 1A.", "anchors": []},
        ]
    )
    assert chosen_first_figure_token == "Fig. 1"
    assert chosen_quant_statement == "We surveyed 834 studies in the ClinicalTrials.gov database and explored biomaterial types, their initiation points, and durations in clinical trials."
    assert quant_candidates[0]["statement"] == chosen_quant_statement
    assert "timeline depicts" not in answer.lower()


def test_quant_extraction_rejects_speculative_late_page_text_without_numbers():
    answer, quant_candidates, chosen_quant_statement, chosen_first_figure_token, _ = extract_quant_plus_figure_answer(
        [
            {"page": 15, "chunk_idx": 201, "block_type": "paragraph", "text_for_embed": "Likely to yield positive results for clinical translation and patient care. As researchers continue to innovate with biomaterials, the distribution of clinical trials will change accordingly to focus on these future biomaterials."},
            {"page": 3, "chunk_idx": 41, "block_type": "paragraph", "text_for_embed": "We surveyed 834 studies in the ClinicalTrials.gov database and explored biomaterial types, their initiation points, and durations in clinical trials."},
            {"page": 1, "chunk_idx": 20, "block_type": "paragraph", "text_for_embed": "A wide array of biomaterials has found successful applications in various clinical settings (Fig. 1)."},
        ]
    )
    assert chosen_first_figure_token == "Fig. 1"
    assert chosen_quant_statement == "We surveyed 834 studies in the ClinicalTrials.gov database and explored biomaterial types, their initiation points, and durations in clinical trials."
    assert quant_candidates[0]["statement"] == chosen_quant_statement


def test_best_quant_statement_rejects_formula_noise():
    noisy = r"F4 transition equation_inline \mathrm {x} text the full width at half maximum was equation_inline 136 ~ \mathrm { n m }."
    assert extract_best_quant_statement(noisy) == ""


def test_best_quant_statement_keeps_real_result_under_ocr_formula_noise():
    noisy_but_useful = (
        r"The 976- and 808-nm laser powers measured at the objective back aperture were equation_inline 1 . 9 and 192 \mathrm { m W } text, "
        r"corresponding to intensities of equation_inline 0 . 13 and 6 . 5 \mathrm { M W } text, respectively."
    )
    assert "laser powers measured" in extract_best_quant_statement(noisy_but_useful)


def test_presence_check_answer_reports_missing_doi_and_figure_token():
    answer = extract_presence_check_answer(
        "Based only on the parsed document, is an explicit DOI or any figure/table label actually present?",
        [
            {"text_for_embed": "This parsed page contains introductory text only.", "page": 1, "chunk_idx": 0, "anchors": []},
        ],
    )
    assert answer == "DOI: not detected; Figure/table: not detected."


def test_extract_target_figure_token_normalizes_panel_suffix():
    assert extract_target_figure_token("According to Fig. 1E, what values are reported?") == "Fig. 1"


def test_chart_numeric_answer_formats_fwhm_pair():
    answer, candidates, chosen_statement, chosen_token = extract_chart_numeric_answer(
        "According to Fig. 1E, what full width at half maximum (FWHM) is reported for the STED image, and what comparison FWHM is given for excitation-only imaging?",
        [
            {
                "page": 2,
                "chunk_idx": 7,
                "text_for_embed": "Representative excitation and STED images of single UCNPs and corresponding line profiles, where the full width at half maximum values are 461 and 136 nm for excitation-only and STED images, respectively (Fig. 1E).",
                "anchors": ["Fig. 1"],
            }
        ],
    )
    assert answer == "STED-image FWHM: 136 nm; excitation-only FWHM: 461 nm."
    assert chosen_statement
    assert chosen_token == "Fig. 1"
    assert candidates


def test_chart_numeric_alignment_prefers_attribute_matched_sentence_over_same_page_distractor():
    answer, candidates, chosen_statement, chosen_token = extract_chart_numeric_answer(
        "According to Fig. 1E, what full width at half maximum (FWHM) is reported for the STED image, and what comparison FWHM is given for excitation-only imaging?",
        [
            {
                "page": 2,
                "chunk_idx": 7,
                "text_for_embed": "The 976- and 808-nm laser powers measured at the objective back aperture were 1.9 and 192 mW, corresponding to intensities of 0.13 MW cm^-2 and 6.5 MW cm^-2, respectively (Fig. 1E).",
                "anchors": ["Fig. 1"],
            },
            {
                "page": 2,
                "chunk_idx": 8,
                "text_for_embed": "The full width at half maximum (FWHM) of a Gaussian fit to the intensity profile from the resulting image is 136 nm, a notable improvement compared to the 461 nm FWHM obtained from imaging the same single UCNP with only the excitation beam (Fig. 1E).",
                "anchors": ["Fig. 1"],
            },
        ],
    )
    assert chosen_statement.startswith("The full width at half maximum")
    assert answer == "STED-image FWHM: 136 nm; excitation-only FWHM: 461 nm."
    assert chosen_token == "Fig. 1"
    assert candidates[0]["statement"] == chosen_statement


def test_chart_numeric_answer_formats_study_count():
    answer, candidates, chosen_statement, chosen_token = extract_chart_numeric_answer(
        "In document id sciadv.abq0997.PMC466960, according to the Fig. 2 search flowchart, how many studies were initially identified using the predefined search criteria?",
        [
            {
                "page": 3,
                "chunk_idx": 4,
                "text_for_embed": "By searching the ClinicalTrials.gov database with predefined search criteria, a total of 2767 studies were initially identified (Fig. 2).",
                "anchors": ["Fig. 2"],
            }
        ],
    )
    assert answer == "2767 studies."
    assert "2767 studies" in chosen_statement
    assert chosen_token == "Fig. 2"
    assert candidates


def test_chart_numeric_normalization_cleans_parser_unit_artifacts():
    cleaned = _normalize_chart_numeric_text("equation_inline 1 3 6 ~ \\mathrm { n m } text and equation_inline 1 0 \\mathsf { m m } text")
    assert "136 nm" in cleaned
    assert "10 mm" in cleaned
    assert "text" not in cleaned.lower()


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
