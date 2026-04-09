from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_lightweight_import_stubs():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        torch_mod = types.ModuleType("torch")

        class _DummyNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        torch_mod.cuda = SimpleNamespace(is_available=lambda: False)
        torch_mod.device = lambda name: name
        torch_mod.no_grad = _DummyNoGrad
        sys.modules["torch"] = torch_mod
        sys.modules["torch.nn"] = types.ModuleType("torch.nn")
        torch_nn_functional = types.ModuleType("torch.nn.functional")
        torch_nn_functional.normalize = lambda tensor, p=2, dim=-1: tensor
        sys.modules["torch.nn.functional"] = torch_nn_functional

    try:
        import sentence_transformers  # noqa: F401
    except ModuleNotFoundError:
        st_mod = types.ModuleType("sentence_transformers")

        class SentenceTransformer:
            def __init__(self, *_args, **_kwargs):
                pass

            def encode(self, *_args, **_kwargs):
                return []

        st_mod.SentenceTransformer = SentenceTransformer
        sys.modules["sentence_transformers"] = st_mod

    try:
        import igraph  # noqa: F401
    except ModuleNotFoundError:
        igraph_mod = types.ModuleType("igraph")

        class Graph:
            def __init__(self, *args, **kwargs):
                self.vs = []

        igraph_mod.Graph = Graph
        sys.modules["igraph"] = igraph_mod

    try:
        import spacy  # noqa: F401
    except ModuleNotFoundError:
        spacy_mod = types.ModuleType("spacy")

        class _DummySpacyModel:
            def pipe(self, *_args, **_kwargs):
                return []

            def __call__(self, _text):
                return SimpleNamespace(ents=[])

        spacy_mod.load = lambda *_args, **_kwargs: _DummySpacyModel()
        sys.modules["spacy"] = spacy_mod

    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        transformers_mod = types.ModuleType("transformers")

        class _DummyProcessor:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def __call__(self, *_args, **_kwargs):
                return SimpleNamespace(to=lambda _device: {})

        class _DummyModel:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def to(self, _device):
                return self

            def eval(self):
                return self

        transformers_mod.CLIPProcessor = _DummyProcessor
        transformers_mod.CLIPModel = _DummyModel
        sys.modules["transformers"] = transformers_mod

    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        pandas_mod = types.ModuleType("pandas")

        class DataFrame:
            def __init__(self, data):
                self.data = data

            def to_parquet(self, *_args, **_kwargs):
                return None

        pandas_mod.DataFrame = DataFrame
        pandas_mod.read_parquet = lambda *_args, **_kwargs: DataFrame({})
        sys.modules["pandas"] = pandas_mod

    try:
        import httpx  # noqa: F401
    except ModuleNotFoundError:
        httpx_mod = types.ModuleType("httpx")
        httpx_mod.Client = lambda *_args, **_kwargs: SimpleNamespace()
        sys.modules["httpx"] = httpx_mod

    try:
        import openai  # noqa: F401
    except ModuleNotFoundError:
        openai_mod = types.ModuleType("openai")

        class RateLimitError(Exception):
            pass

        class OpenAI:
            def __init__(self, *_args, **_kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **_kwargs: SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content="stub"))]
                        )
                    )
                )

        openai_mod.RateLimitError = RateLimitError
        openai_mod.OpenAI = OpenAI
        sys.modules["openai"] = openai_mod

    try:
        import tqdm  # noqa: F401
    except ModuleNotFoundError:
        tqdm_mod = types.ModuleType("tqdm")
        tqdm_mod.tqdm = lambda iterable=None, *args, **kwargs: iterable if iterable is not None else []
        sys.modules["tqdm"] = tqdm_mod


_install_lightweight_import_stubs()

import run
from src.LinearRAG import LinearRAG
from src.utils import LLM_Model


def test_prepare_passages_for_indexing_obeys_explicit_limit_only():
    passages = [
        "0:base",
        {"passage_id": "doc:p1", "doc_id": "doc", "page": 1},
        "pdf:file.pdf:chunk0:pdf-text",
    ]
    assert run.prepare_passages_for_indexing(passages, -1) == passages
    assert run.prepare_passages_for_indexing(passages, 2) == passages[:2]


def test_prepare_passages_for_indexing_raises_when_pdf_limit_removes_structured_passages():
    passages = [
        "0:legacy",
        "1:legacy",
        {"passage_id": "doc:p1", "doc_id": "doc", "page": 1},
    ]
    with pytest.raises(ValueError, match="structured PDF passages"):
        run.prepare_passages_for_indexing(passages, 2, use_pdf_retrieval=True)


def test_validate_arguments_rejects_ambiguous_zero_values():
    args = SimpleNamespace(
        llm_model="GLM-4.6V-Flash",
        max_workers=1,
        max_qa_images=1,
        retrieval_top_k_image=1,
        image_ratio=0.5,
        debug_num_questions=0,
        debug_num_passages=-1,
        use_pdf_retrieval=False,
        pdf_paths_file=None,
    )
    with pytest.raises(ValueError, match="debug_num_questions"):
        run.validate_arguments(args)


def test_build_qa_messages_raises_for_missing_image_path():
    model = LLM_Model("GLM-4.6V-Flash")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        model.build_qa_messages(
            system_prompt="system",
            prompt_user_text="question",
            image_paths=[str(ROOT / "missing_image.png")],
        )


def test_multimodal_text_fallback_only_triggers_for_schema_errors():
    model = LLM_Model("GLM-4.6V-Flash")
    assert model._should_fallback_to_text_only(ValueError("unsupported content type image_url")) is True
    assert model._should_fallback_to_text_only(RuntimeError("Connection error")) is False


def test_build_image_to_passage_mapping_and_overlap_debug():
    rag = LinearRAG.__new__(LinearRAG)
    rag.config = SimpleNamespace(use_image_retrieval=True, retrieval_top_k_image=5)
    rag.image_metadata_by_hash = {
        "image-a": {"doc_id": "doc1", "page": 2, "path": "images/doc1_p2.png"},
    }
    rag.pdf_structure_state = {
        "page_nodes": {
            "page::doc1::2": {
                "doc_id": "doc1",
                "page": 2,
                "passage_hash_ids": ["p2", "p3"],
                "image_hash_ids": ["image-a"],
            }
        }
    }
    mapping = LinearRAG.build_image_to_passage_mapping(rag)
    assert mapping["image-a"]["related_passage_ids"] == ["p2", "p3"]
    rag.image_to_passage_mapping = mapping
    enriched = LinearRAG.enrich_retrieved_images(rag, [{"hash_id": "image-a", "path": "images/doc1_p2.png", "score": 0.7}])
    assert enriched[0]["page"] == 2
    overlap = LinearRAG.build_image_passage_overlap(rag, enriched, ["p1", "p3"])
    assert overlap[0]["selected_passage_overlap"] == ["p3"]


def test_build_image_to_passage_mapping_raises_when_image_has_no_page_link():
    rag = LinearRAG.__new__(LinearRAG)
    rag.config = SimpleNamespace(use_image_retrieval=True, retrieval_top_k_image=5)
    rag.image_metadata_by_hash = {
        "image-a": {"doc_id": "doc1", "page": 2, "path": "images/doc1_p2.png"},
    }
    rag.pdf_structure_state = {"page_nodes": {}}
    with pytest.raises(KeyError, match="Missing page/passage mapping"):
        LinearRAG.build_image_to_passage_mapping(rag)


class _DummyEmbedder:
    def encode(self, *_args, **_kwargs):
        return np.array([1.0, 0.0], dtype=float)


def _make_minimal_rag_for_retrieve():
    rag = LinearRAG.__new__(LinearRAG)
    rag.config = SimpleNamespace(
        use_image_retrieval=False,
        use_vectorized_retrieval=False,
        router_semantic_threshold=0.42,
        embedding_model=_DummyEmbedder(),
        batch_size=1,
        retrieval_top_k=1,
        retrieval_top_k_image=1,
    )
    empty_store = SimpleNamespace(hash_id_to_text={}, embeddings=[], hash_ids=[], texts=[])
    rag.entity_embedding_store = empty_store
    rag.sentence_embedding_store = empty_store
    rag.passage_embedding_store = SimpleNamespace(
        hash_id_to_text={"p1": "retrieved passage"},
        embeddings=[[1.0, 0.0]],
        hash_ids=["p1"],
        texts=["retrieved passage"],
    )
    rag.graph = SimpleNamespace(vs=[])
    rag.semantic_router = None
    rag.passage_metadata_by_hash = {}
    rag.image_metadata_by_hash = {}
    rag.doc_to_passage_hash_ids = {}
    rag.doc_to_image_hash_ids = {}
    rag.pdf_structure_state = {}
    rag.get_seed_entities = lambda _question: ([], [], [], [])
    rag.run_scope_planner = lambda *_args, **_kwargs: (
        [],
        [],
        {
            "candidate_count": 0,
            "selected_evidence_ids": [],
            "selected_slot_assignments": {},
            "scope_score_breakdown": {},
            "selected_evidence": [],
            "structural_feature_hits": {},
            "candidate_reliability": {},
            "slot_reliability": {},
            "plan_reliability": 0.0,
            "fallback_used": False,
            "fallback_reason": "",
        },
    )
    rag.dense_passage_retrieval = lambda *_args, **_kwargs: ([0], [0.95])
    return rag


def test_retrieve_prefers_dataset_question_type_over_heuristic():
    rag = _make_minimal_rag_for_retrieve()
    results = rag.retrieve(
        [
            {
                "question": "Please answer this generic request.",
                "question_type": "summary",
                "answer": "gold",
            }
        ]
    )
    assert results[0]["question_type"] == "summary"


def test_retrieve_falls_back_to_heuristic_question_type_when_missing():
    rag = _make_minimal_rag_for_retrieve()
    results = rag.retrieve(
        [
            {
                "question": "What is the exact full paper title?",
                "answer": "gold",
            }
        ]
    )
    assert results[0]["question_type"] == "title"
