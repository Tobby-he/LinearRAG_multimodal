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
    route_question_hybrid,
)
from src.semantic_router import SemanticPrototypeRouter


class FakeEmbedder:
    vocab = [
        "title", "article", "paper", "full",
        "figure", "table", "caption", "token", "earliest", "non-supplementary", "main-paper",
        "summary", "main", "contribution", "study", "sentence",
        "quantitative", "numeric", "finding", "statement",
        "doi",
    ]

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            t = (text or "").lower()
            vec = []
            for token in self.vocab:
                token_norm = token.replace("-", " ")
                vec.append(float(token_norm in t))
            vectors.append(vec)
        return vectors


def build_router():
    return SemanticPrototypeRouter(FakeEmbedder(), confidence_threshold=0.25)


def test_semantic_router_basic_classification_for_representative_phrasings():
    router = build_router()
    assert router.predict("what is the title of this paper?")["predicted_route"] == TITLE_ROUTE
    assert router.predict("give the full article title")["predicted_route"] == TITLE_ROUTE
    assert router.predict("which figure or table appears first in the main paper?")["predicted_route"] == FIGURE_ROUTE
    assert router.predict("what is the earliest non-supplementary figure/table label?")["predicted_route"] == FIGURE_ROUTE
    assert router.predict("state the main contribution of this paper in one sentence")["predicted_route"] == SUMMARY_ROUTE
    assert router.predict("what does this study mainly contribute?")["predicted_route"] == SUMMARY_ROUTE
    assert router.predict("provide one key numeric finding together with the first main-paper figure/table token")["predicted_route"] == QUANT_ROUTE
    assert router.predict("return one quantitative statement and the earliest main-paper figure/table token")["predicted_route"] == QUANT_ROUTE


def test_hybrid_router_prefers_rule_when_strong_rule_match_exists():
    router = build_router()
    result = route_question_hybrid(
        "What is the exact full paper title?",
        semantic_router=router,
        semantic_threshold=0.25,
    )
    assert result["predicted_route"] == TITLE_ROUTE
    assert result["route_source"] == "rule"


def test_hybrid_router_uses_semantic_when_rules_do_not_match():
    router = build_router()
    result = route_question_hybrid(
        "Give the full article title.",
        semantic_router=router,
        semantic_threshold=0.25,
    )
    assert result["predicted_route"] == TITLE_ROUTE
    assert result["route_source"] == "semantic"
    assert result["matched_prototypes"]


def test_hybrid_router_low_confidence_falls_back_to_general():
    router = build_router()
    result = route_question_hybrid(
        "List the author affiliations.",
        semantic_router=router,
        semantic_threshold=0.8,
    )
    assert result["predicted_route"] == GENERAL_ROUTE
    assert result["route_source"] == "fallback"
