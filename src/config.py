from dataclasses import dataclass
from src.utils import LLM_Model
@dataclass
class LinearRAGConfig:
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    vision_embedding_model: str = "openai/clip-vit-base-patch32"  # Default vision model
    llm_model: LLM_Model = None
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_trf"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    use_vectorized_retrieval: bool = False  # True for vectorized matrix computation, False for BFS iteration
    use_image_retrieval: bool = False # Enable image retrieval
    use_pdf_retrieval: bool = False # Enable PDF retrieval
    retrieval_top_k_image: int = 5
    image_ratio: float = 0.5
    max_qa_images: int = 3
    router_semantic_threshold: float = 0.42
    router_aggregation_mode: str = "max"
