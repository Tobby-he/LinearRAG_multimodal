import argparse
import json
import os
import warnings
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer

from src.LinearRAG import LinearRAG
from src.config import LinearRAGConfig
from src.evaluate import Evaluator
from src.pdf_parser import load_pdf_documents
from src.utils import LLM_Model, setup_logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")


def split_text_into_chunks(text, chunk_size=1200, overlap=150):
    if not text:
        return []
    if chunk_size <= overlap:
        overlap = 0
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf", help="The spacy model to use")
    parser.add_argument("--embedding_model", type=str, default="model/all-mpnet-base-v2", help="The path of embedding model to use")
    parser.add_argument("--dataset_name", type=str, default="novel", help="The dataset to use")
    parser.add_argument("--llm_model", type=str, default="qwen3.5-397b-a17b", help="The LLM model to use")
    parser.add_argument("--max_workers", type=int, default=16, help="The max number of workers to use")
    parser.add_argument("--max_iterations", type=int, default=3, help="The max number of iterations to use")
    parser.add_argument("--iteration_threshold", type=float, default=0.4, help="The threshold for iteration")
    parser.add_argument("--passage_ratio", type=float, default=2, help="The ratio for passage")
    parser.add_argument("--top_k_sentence", type=int, default=3, help="The top k sentence to use")
    parser.add_argument("--use_vectorized_retrieval", action="store_true", help="Use vectorized matrix-based retrieval instead of BFS iteration")
    parser.add_argument("--image_paths_file", type=str, default=None, help="Path to a JSON file containing image paths for multimodal retrieval.")
    parser.add_argument("--use_image_retrieval", action="store_true", help="Enable multimodal image retrieval.")
    parser.add_argument("--pdf_paths_file", type=str, default=None, help="Path to a JSON file containing PDF file paths for multimodal retrieval.")
    parser.add_argument("--use_pdf_retrieval", action="store_true", help="Enable multimodal PDF retrieval.")
    parser.add_argument("--max_qa_images", type=int, default=3, help="Max number of retrieved images injected into QA model input.")
    return parser.parse_args()


def load_dataset(dataset_name, args):
    questions_path = f"dataset/{dataset_name}/questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    chunks_path = f"dataset/{dataset_name}/chunks.json"
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        chunks = []
    passages = [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]

    images_data = []
    if args.image_paths_file:
        full_image_path = os.path.join(os.path.dirname(questions_path), args.image_paths_file)
        if os.path.exists(full_image_path):
            with open(full_image_path, "r", encoding="utf-8") as f:
                images_data = json.load(f)
            print(f"Loaded {len(images_data)} image records.")

    pdf_data = []
    pdf_images = []
    if args.use_pdf_retrieval and args.pdf_paths_file:
        if os.path.isabs(args.pdf_paths_file):
            full_pdf_path_file = args.pdf_paths_file
        else:
            full_pdf_path_file = os.path.join(os.path.dirname(questions_path), args.pdf_paths_file)

        pdf_data, pdf_images = load_pdf_documents(full_pdf_path_file)
        for doc in pdf_data:
            for i, chunk in enumerate(split_text_into_chunks(doc["content"])):
                passages.append(f"pdf:{os.path.basename(doc['path'])}:chunk{i}:{chunk}")
        print(f"Loaded and parsed {len(pdf_data)} PDF docs, extracted {len(pdf_images)} images.")

    return questions, passages, images_data, pdf_data, pdf_images


def load_embedding_model(embedding_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using embedding device: {device}")
    return SentenceTransformer(embedding_model, device=device)


def main():
    print("Program started.")
    time_now = datetime.now()
    time_str = time_now.strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_arguments()

    print(f"Loading embedding model: {args.embedding_model}")
    embedding_model = load_embedding_model(args.embedding_model)

    print(f"Loading dataset: {args.dataset_name}")
    questions, passages, images_data, pdf_data, pdf_images = load_dataset(args.dataset_name, args)

    if args.use_pdf_retrieval:
        images_data.extend(pdf_images)
    effective_use_image_retrieval = args.use_image_retrieval or args.use_pdf_retrieval

    # Keep original quick-debug behavior.
    questions = questions[:5]
    if args.use_pdf_retrieval and len(pdf_data) > 0:
        pdf_passages = [p for p in passages if isinstance(p, str) and p.startswith("pdf:")]
        passages = passages[:100] + pdf_passages
    else:
        passages = passages[:100]

    print(f"Debug mode: {len(questions)} questions, {len(passages)} passages.")
    if effective_use_image_retrieval and len(images_data) > 0:
        print(f"Image retrieval enabled with {len(images_data)} images.")

    setup_logging(f"results/{args.dataset_name}/{time_str}/log.txt")
    llm_model = LLM_Model(args.llm_model)

    config = LinearRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
        use_image_retrieval=effective_use_image_retrieval,
        use_pdf_retrieval=args.use_pdf_retrieval,
        max_qa_images=args.max_qa_images,
    )
    rag_model = LinearRAG(global_config=config)

    rag_model.index(passages, images_data if effective_use_image_retrieval else [], pdf_data if args.use_pdf_retrieval else [])
    results = rag_model.qa(questions)

    os.makedirs(f"results/{args.dataset_name}/{time_str}", exist_ok=True)
    predictions_path = f"results/{args.dataset_name}/{time_str}/predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    evaluator = Evaluator(llm_model=llm_model, predictions_path=predictions_path)
    evaluator.evaluate(max_workers=args.max_workers)


if __name__ == "__main__":
    main()
