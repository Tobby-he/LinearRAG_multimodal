from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from src.semantic_router import GENERAL_ROUTE


@dataclass
class EvidencePlan:
    task_type: str
    evidence_slots: List[str]
    structural_constraints: List[str]
    budget: str
    answer_format: str
    target_doc_id: str = ""
    router_debug: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


PLAN_LIBRARY = {
    "title": {
        "evidence_slots": ["TITLE"],
        "structural_constraints": ["front_page", "title_like_block"],
        "budget": "small",
        "answer_format": "exact_string",
    },
    "first_figure_token": {
        "evidence_slots": ["FIGURE_TOKEN"],
        "structural_constraints": ["main_paper_only", "earliest_eligible"],
        "budget": "small",
        "answer_format": "token_only",
    },
    "summary": {
        "evidence_slots": ["CONTRIBUTION_SENTENCE"],
        "structural_constraints": ["abstract_or_intro", "front_matter_preferred"],
        "budget": "moderate",
        "answer_format": "one_sentence",
    },
    "quant_plus_first_figure": {
        "evidence_slots": ["FIGURE_TOKEN", "QUANT_STATEMENT"],
        "structural_constraints": ["main_paper_only", "earliest_eligible", "numeric_statement_required"],
        "budget": "moderate",
        "answer_format": "structured_pair",
    },
    "chart_numeric": {
        "evidence_slots": ["TARGET_FIGURE_TOKEN", "QUANT_STATEMENT"],
        "structural_constraints": ["main_paper_only", "target_figure_match", "numeric_statement_required"],
        "budget": "moderate",
        "answer_format": "chart_numeric",
    },
    GENERAL_ROUTE: {
        "evidence_slots": ["GENERAL_EVIDENCE"],
        "structural_constraints": [],
        "budget": "default",
        "answer_format": "default",
    },
}


def build_evidence_plan(question, predicted_route, router_debug, target_doc_id=None):
    route = predicted_route if predicted_route in PLAN_LIBRARY else GENERAL_ROUTE
    template = PLAN_LIBRARY[route]
    return EvidencePlan(
        task_type=route,
        evidence_slots=list(template["evidence_slots"]),
        structural_constraints=list(template["structural_constraints"]),
        budget=template["budget"],
        answer_format=template["answer_format"],
        target_doc_id=target_doc_id or "",
        router_debug=dict(router_debug or {}),
    )
