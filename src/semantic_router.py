import math
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import numpy as np

TITLE_ROUTE = "title"
FIGURE_ROUTE = "first_figure_token"
QUANT_ROUTE = "quant_plus_first_figure"
SUMMARY_ROUTE = "summary"
GENERAL_ROUTE = "general"
CHART_NUMERIC_ROUTE = "chart_numeric"

DEFAULT_ROUTE_PROTOTYPES = {
    TITLE_ROUTE: [
        "What is the exact full paper title?",
        "What is the title of this paper?",
        "Give the full article title.",
        "Please give the complete paper title.",
        "What is the official title of this article?",
        "Return the complete article title.",
    ],
    FIGURE_ROUTE: [
        "What is the first figure/table caption token mentioned?",
        "Which figure or table appears first in the main paper?",
        "What is the earliest non-supplementary figure/table label?",
        "Return only the first main-paper figure/table token.",
    ],
    SUMMARY_ROUTE: [
        "Summarize the main contribution in one sentence.",
        "State the main contribution of this paper in one sentence.",
        "What does this study mainly contribute? Answer in one sentence.",
    ],
    QUANT_ROUTE: [
        "Combine one key quantitative statement with the first figure/table caption token.",
        "Provide one key numeric finding together with the first main-paper figure/table token.",
        "Return one quantitative statement and the earliest main-paper figure/table token.",
        "Report one quantitative finding from the study and also name the earliest figure/table label in the main article.",
        "Give one important numeric result from the paper together with the first main-paper figure/table label.",
        "Provide a numeric result and the earliest figure/table label from the main article.",
    ],
    CHART_NUMERIC_ROUTE: [
        "According to Fig. 1, what numeric value is reported?",
        "According to Fig. 2, how many studies were identified?",
        "From Fig. 1E, what values are reported for the compared measurements?",
        "Using the specified figure or table, return only the requested numeric value.",
    ],
    GENERAL_ROUTE: [
        "Does the parsed document explicitly contain a DOI or a figure/table token?",
        "What does this paper say?",
        "Answer the question using the retrieved evidence.",
    ],
}


def _to_matrix(vectors) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class SemanticPrototypeRouter:
    def __init__(
        self,
        embedding_model,
        route_prototypes: Dict[str, List[str]] | None = None,
        aggregation_mode: str = "max",
        confidence_threshold: float = 0.42,
        top_k_scores: int = 3,
    ):
        self.embedding_model = embedding_model
        self.route_prototypes = route_prototypes or DEFAULT_ROUTE_PROTOTYPES
        self.aggregation_mode = aggregation_mode
        self.confidence_threshold = confidence_threshold
        self.top_k_scores = top_k_scores
        self._prototype_texts: List[str] = []
        self._prototype_routes: List[str] = []
        self._route_to_proto_indices: Dict[str, List[int]] = {}
        self._build_prototype_index()

    def _build_prototype_index(self):
        for route, questions in self.route_prototypes.items():
            self._route_to_proto_indices[route] = []
            for question in questions:
                idx = len(self._prototype_texts)
                self._prototype_texts.append(question)
                self._prototype_routes.append(route)
                self._route_to_proto_indices[route].append(idx)
        self._prototype_embeddings = _to_matrix(self.embedding_model.encode(self._prototype_texts, normalize_embeddings=True, show_progress_bar=False))

    def _aggregate(self, scores: Iterable[float]) -> float:
        values = list(scores)
        if not values:
            return -1.0
        if self.aggregation_mode == "top2_mean":
            values = sorted(values, reverse=True)[:2]
            return float(sum(values) / len(values))
        return float(max(values))

    @lru_cache(maxsize=2048)
    def predict(self, question: str) -> dict:
        q = (question or "").strip()
        if not q:
            return {
                "predicted_route": GENERAL_ROUTE,
                "route_confidence": 0.0,
                "top_route_scores": {GENERAL_ROUTE: 0.0},
                "matched_prototypes": [],
            }
        q_emb = _to_matrix(self.embedding_model.encode([q], normalize_embeddings=True, show_progress_bar=False))[0]
        similarities = np.dot(self._prototype_embeddings, q_emb)
        route_scores = {}
        for route, indices in self._route_to_proto_indices.items():
            route_scores[route] = self._aggregate(similarities[idx] for idx in indices)
        sorted_routes = sorted(route_scores.items(), key=lambda x: x[1], reverse=True)
        best_route, best_score = sorted_routes[0]
        top_proto_idx = sorted(range(len(self._prototype_texts)), key=lambda i: similarities[i], reverse=True)[: self.top_k_scores]
        matched_prototypes = [
            {
                "route": self._prototype_routes[i],
                "prototype": self._prototype_texts[i],
                "score": float(similarities[i]),
            }
            for i in top_proto_idx
        ]
        return {
            "predicted_route": best_route,
            "route_confidence": float(best_score),
            "top_route_scores": {route: float(score) for route, score in sorted_routes},
            "matched_prototypes": matched_prototypes,
        }
