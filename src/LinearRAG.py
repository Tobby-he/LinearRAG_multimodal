from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize, compute_mdhash_id
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.ner import SpacyNER
import igraph as ig
import re
import logging
import torch
from src.vision_encoder import VisionEncoder
from src.doc_aware import (
    build_doc_to_hash_ids,
    detect_weak_parser_coverage,
    infer_question_type,
    normalize_doc_id,
    parse_doc_id_from_question,
    resolve_question_type,
    resolve_allowed_doc_hash_ids,
)
from src.evidence_candidates import collect_plan_candidates
from src.evidence_plan import build_evidence_plan
from src.pdf_structure_graph import (
    build_pdf_structure_state,
    infer_page_from_image_path,
)
from src.m3_lite import (
    FIGURE_ROUTE,
    GENERAL_ROUTE,
    QUANT_ROUTE,
    SUMMARY_ROUTE,
    TITLE_ROUTE,
    extract_best_quant_statement,
    extract_chart_numeric_answer,
    extract_first_figure_answer,
    extract_presence_check_answer,
    extract_quant_plus_figure_answer,
    extract_summary_answer,
    extract_summary_candidates,
    extract_title_answer,
    route_question_hybrid,
    select_doc_passage_items,
)
from src.semantic_router import CHART_NUMERIC_ROUTE, SemanticPrototypeRouter
from src.scope_planner import plan_select_evidence
from src.task_eval import extract_gold_payload
logger = logging.getLogger(__name__)


class LinearRAG:
    def __init__(self, global_config):
        self.config = global_config
        logger.info(f"Initializing LinearRAG with config: {self.config}")
        retrieval_method = "Vectorized Matrix-based" if self.config.use_vectorized_retrieval else "BFS Iteration"
        logger.info(f"Using retrieval method: {retrieval_method}")
        
        # Setup device for GPU acceleration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.config.use_vectorized_retrieval:
            logger.info(f"Using device: {self.device} for vectorized retrieval")
        
        self.dataset_name = global_config.dataset_name
        
        # 初始化 VisionEncoder (如果需要)
        if self.config.use_image_retrieval:
            self.vision_encoder = VisionEncoder(self.config.vision_embedding_model, device=str(self.device))
        
        self.load_embedding_store()
        self.llm_model = self.config.llm_model
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        self.graph = ig.Graph(directed=False)
        self.passage_metadata_by_hash = {}
        self.image_metadata_by_hash = {}
        self.doc_to_passage_hash_ids = {}
        self.doc_to_image_hash_ids = {}
        self.image_to_passage_mapping = {}
        self.pdf_structure_state = {}
        self.semantic_router = SemanticPrototypeRouter(
            self.config.embedding_model,
            aggregation_mode=self.config.router_aggregation_mode,
            confidence_threshold=self.config.router_semantic_threshold,
        )

    def load_embedding_store(self):
        self.passage_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "passage_embedding.parquet"), batch_size=self.config.batch_size, namespace="passage")
        self.entity_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "entity_embedding.parquet"), batch_size=self.config.batch_size, namespace="entity")
        self.sentence_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "sentence_embedding.parquet"), batch_size=self.config.batch_size, namespace="sentence")
        if self.config.use_image_retrieval:
            # 关键：图像 Store 必须使用 vision_encoder
            self.image_embedding_store = EmbeddingStore(self.vision_encoder, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "image_embedding.parquet"), batch_size=self.config.batch_size, namespace="image")

    def load_existing_data(self,passage_hash_ids):
        self.ner_results_path = os.path.join(self.config.working_dir,self.dataset_name, "ner_results.json")
        if os.path.exists(self.ner_results_path):
            existing_ner_reuslts = json.load(open(self.ner_results_path))
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def qa(self, questions):
        retrieval_results = self.retrieve(questions)
        system_prompt = (
            "You are a precise QA assistant. "
            "Use only the retrieved evidence to answer. "
            "Output format must be exactly: Answer: <final answer>. "
            "Do not output reasoning steps."
        )
        all_messages = []
        llm_result_indices = []
        for idx, retrieval_result in enumerate(retrieval_results):
            if retrieval_result.get("direct_answer") is not None:
                continue
            question = retrieval_result["question"]
            sorted_passage = retrieval_result["sorted_passage"]
            retrieved_images = retrieval_result.get("retrieved_images", [])
            prompt_user = """"""
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            if retrieved_images:
                prompt_user += "Retrieved Images:\n"
                for image_item in retrieved_images:
                    prompt_user += f"- {image_item['path']} (score={image_item['score']:.4f})\n"
            prompt_user += f"Question: {question}\nPlease answer in one concise sentence.\n"
            image_paths = [item["path"] for item in retrieved_images[: self.config.max_qa_images]]
            messages = self.llm_model.build_qa_messages(
                system_prompt=system_prompt,
                prompt_user_text=prompt_user,
                image_paths=image_paths,
            )
            all_messages.append(messages)
            llm_result_indices.append(idx)
        all_qa_results = []
        if all_messages:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                all_qa_results = list(tqdm(
                    executor.map(self.llm_model.infer, all_messages),
                    total=len(all_messages),
                    desc="QA Reading (Parallel)"
                ))

        llm_idx_to_result = {}
        for qa_result, retrieval_idx in zip(all_qa_results, llm_result_indices):
            llm_idx_to_result[retrieval_idx] = qa_result

        for idx, question_info in enumerate(retrieval_results):
            if question_info.get("direct_answer") is not None:
                question_info["pred_answer"] = question_info["direct_answer"]
                continue
            qa_result = llm_idx_to_result[idx]
            qa_text = qa_result.strip() if isinstance(qa_result, str) else str(qa_result)
            m = re.search(r"(?:^|\\n)\\s*(?:Answer|答案)\\s*[:：]\\s*(.+)", qa_text, flags=re.IGNORECASE | re.DOTALL)
            pred_ans = m.group(1).strip() if m else qa_text
            question_info["pred_answer"] = pred_ans
        return retrieval_results
        
    def retrieve(self, questions):
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        if self.config.use_image_retrieval and hasattr(self, "image_embedding_store"):
            self.image_hash_ids = list(self.image_embedding_store.hash_id_to_text.keys())
            self.image_embeddings = np.array(self.image_embedding_store.embeddings)
        else:
            self.image_hash_ids = []
            self.image_embeddings = np.array([])
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()}

        # Precompute sparse matrices for vectorized retrieval if needed
        if self.config.use_vectorized_retrieval:
            logger.info("Precomputing sparse adjacency matrices for vectorized retrieval...")
            self._precompute_sparse_matrices()
            e2s_shape = self.entity_to_sentence_sparse.shape
            s2e_shape = self.sentence_to_entity_sparse.shape
            e2s_nnz = self.entity_to_sentence_sparse._nnz()
            s2e_nnz = self.sentence_to_entity_sparse._nnz()
            logger.info(f"Matrices built: Entity-Sentence {e2s_shape}, Sentence-Entity {s2e_shape}")
            logger.info(f"E2S Sparsity: {(1 - e2s_nnz / (e2s_shape[0] * e2s_shape[1])) * 100:.2f}% (nnz={e2s_nnz})")
            logger.info(f"S2E Sparsity: {(1 - s2e_nnz / (s2e_shape[0] * s2e_shape[1])) * 100:.2f}% (nnz={s2e_nnz})")
            logger.info(f"Device: {self.device}")

        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            raw_question = question_info["question"]
            target_doc_id, question = parse_doc_id_from_question(raw_question)
            question_type = resolve_question_type(question_info, raw_question, logger=logger)
            route_info = route_question_hybrid(
                question,
                semantic_router=self.semantic_router,
                semantic_threshold=self.config.router_semantic_threshold,
            )
            predicted_route = route_info["predicted_route"]
            route_source = route_info["route_source"]
            route_confidence = route_info["route_confidence"]
            top_route_scores = route_info["top_route_scores"]
            matched_prototypes = route_info["matched_prototypes"]
            evidence_plan = build_evidence_plan(
                question,
                predicted_route,
                route_info,
                target_doc_id=target_doc_id,
            )
            logger.info(
                "router_debug question=%s predicted_route=%s route_source=%s route_confidence=%.4f top_route_scores=%s matched_prototypes=%s",
                raw_question,
                predicted_route,
                route_source,
                route_confidence,
                dict(list(top_route_scores.items())[:3]),
                matched_prototypes[:3],
            )
            page_nodes_for_doc = self.pdf_structure_state.get("doc_to_page_nodes", {}).get(target_doc_id, []) if target_doc_id else []
            structure_lookup_ok = bool(target_doc_id and page_nodes_for_doc)
            logger.info(
                "structure_lookup target_doc_id=%s success=%s pages=%d has_page_nodes=%s",
                target_doc_id,
                structure_lookup_ok,
                len(page_nodes_for_doc),
                bool(page_nodes_for_doc),
            )
            parser_coverage_warning = ""
            if target_doc_id:
                weak_coverage, weak_reason = detect_weak_parser_coverage(
                    target_doc_id,
                    self.pdf_structure_state.get("doc_to_page_nodes", {}),
                    self.doc_to_image_hash_ids,
                )
                if weak_coverage:
                    parser_coverage_warning = weak_reason
                    logger.warning("parser_coverage target_doc_id=%s %s", target_doc_id, weak_reason)
            question_embedding = self.config.embedding_model.encode(question,normalize_embeddings=True,show_progress_bar=False,batch_size=self.config.batch_size)
            seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores = self.get_seed_entities(question)
            allowed_passage_ids = resolve_allowed_doc_hash_ids(target_doc_id, self.passage_metadata_by_hash, self.doc_to_passage_hash_ids)
            allowed_image_ids = resolve_allowed_doc_hash_ids(target_doc_id, self.image_metadata_by_hash, self.doc_to_image_hash_ids)
            used_route = "dense_global"
            evidence_pages = []
            title_candidates = []
            figure_token_candidates = []
            summary_candidates = []
            quant_candidates = []
            chart_numeric_candidates = []
            chosen_quant_statement = ""
            chosen_first_figure_token = ""
            direct_answer = None
            image_retrieval_skipped = False
            image_retrieval_skipped_reason = ""
            scope_retrieved_images = []
            scope_selected_evidence = []
            selected_evidence_ids = []
            selected_slot_assignments = {}
            scope_score_breakdown = {}
            structural_feature_hits = {}
            candidate_reliability = {}
            slot_reliability = {}
            plan_reliability = 0.0
            candidate_count = 0
            fallback_used = False
            fallback_reason = ""

            scope_passage_items, scope_image_items, scope_debug = self.run_scope_planner(
                question,
                target_doc_id,
                evidence_plan,
                question_info,
            )
            logger.info(
                "scope_debug target_doc_id=%s task=%s candidate_count=%d selected_ids=%s fallback_used=%s plan_reliability=%.3f fallback_reason=%s",
                target_doc_id,
                evidence_plan.task_type,
                scope_debug.get("candidate_count", 0),
                scope_debug.get("selected_evidence_ids", [])[:5],
                scope_debug.get("fallback_used", False),
                float(scope_debug.get("plan_reliability", 0.0)),
                scope_debug.get("fallback_reason", ""),
            )
            candidate_count = scope_debug.get("candidate_count", 0)
            selected_evidence_ids = scope_debug.get("selected_evidence_ids", [])
            selected_slot_assignments = scope_debug.get("selected_slot_assignments", {})
            scope_score_breakdown = scope_debug.get("scope_score_breakdown", {})
            scope_selected_evidence = scope_debug.get("selected_evidence", [])
            structural_feature_hits = scope_debug.get("structural_feature_hits", {})
            candidate_reliability = scope_debug.get("candidate_reliability", {})
            slot_reliability = scope_debug.get("slot_reliability", {})
            plan_reliability = float(scope_debug.get("plan_reliability", 0.0))
            fallback_used = scope_debug.get("fallback_used", False)
            fallback_reason = scope_debug.get("fallback_reason", "")

            if target_doc_id and predicted_route == TITLE_ROUTE and scope_passage_items:
                direct_answer, title_candidates = extract_title_answer(scope_passage_items)
                selected_items = scope_passage_items[: self.config.retrieval_top_k]
                final_passage_hash_ids = [x["hash_id"] for x in selected_items]
                final_passage_scores = self.build_scope_scores(selected_items, scope_score_breakdown)
                evidence_pages = sorted({x.get("page") for x in selected_items if isinstance(x.get("page"), int)})
                used_route = "scope_title"
            elif target_doc_id and predicted_route == FIGURE_ROUTE and scope_passage_items:
                direct_answer, figure_token_candidates = extract_first_figure_answer(scope_passage_items)
                chosen_first_figure_token = direct_answer
                selected_items = scope_passage_items[: self.config.retrieval_top_k]
                final_passage_hash_ids = [x["hash_id"] for x in selected_items]
                final_passage_scores = self.build_scope_scores(selected_items, scope_score_breakdown)
                evidence_pages = sorted({x.get("page") for x in selected_items if isinstance(x.get("page"), int)})
                used_route = "scope_first_figure_token"
            elif target_doc_id and predicted_route == QUANT_ROUTE and scope_passage_items:
                direct_answer, quant_candidates, chosen_quant_statement, chosen_first_figure_token, figure_token_candidates = extract_quant_plus_figure_answer(scope_passage_items)
                candidate_by_id = {item.get("candidate_id"): item for item in scope_selected_evidence}
                quant_candidate = candidate_by_id.get(selected_slot_assignments.get("QUANT_STATEMENT"))
                figure_candidate = candidate_by_id.get(selected_slot_assignments.get("FIGURE_TOKEN"))
                if quant_candidate:
                    slot_quant_statement = extract_best_quant_statement(quant_candidate.get("text", ""))
                    if slot_quant_statement:
                        chosen_quant_statement = slot_quant_statement
                if figure_candidate and figure_candidate.get("text"):
                    chosen_first_figure_token = figure_candidate.get("text")
                if chosen_quant_statement or chosen_first_figure_token:
                    if chosen_quant_statement and chosen_first_figure_token:
                        direct_answer = f"Key statement: {chosen_quant_statement} First figure/table: {chosen_first_figure_token}"
                    elif chosen_first_figure_token:
                        direct_answer = f"Key statement: not found. First figure/table: {chosen_first_figure_token}"
                    else:
                        direct_answer = f"Key statement: {chosen_quant_statement} First figure/table: not found"
                selected_items = scope_passage_items[: self.config.retrieval_top_k]
                final_passage_hash_ids = [x["hash_id"] for x in selected_items]
                final_passage_scores = self.build_scope_scores(selected_items, scope_score_breakdown)
                evidence_pages = sorted({x.get("page") for x in selected_items if isinstance(x.get("page"), int)})
                used_route = "scope_quant_plus_first_figure"
            elif target_doc_id and predicted_route == CHART_NUMERIC_ROUTE and scope_passage_items:
                direct_answer, chart_numeric_candidates, chosen_quant_statement, chosen_first_figure_token = extract_chart_numeric_answer(
                    question,
                    scope_passage_items,
                )
                candidate_by_id = {item.get("candidate_id"): item for item in scope_selected_evidence}
                target_figure_candidate = candidate_by_id.get(selected_slot_assignments.get("TARGET_FIGURE_TOKEN"))
                if target_figure_candidate and target_figure_candidate.get("text"):
                    chosen_first_figure_token = target_figure_candidate.get("text")
                selected_items = scope_passage_items[: self.config.retrieval_top_k]
                final_passage_hash_ids = [x["hash_id"] for x in selected_items]
                final_passage_scores = self.build_scope_scores(selected_items, scope_score_breakdown)
                evidence_pages = sorted({x.get("page") for x in selected_items if isinstance(x.get("page"), int)})
                used_route = "scope_chart_numeric"
            elif target_doc_id and predicted_route == SUMMARY_ROUTE and scope_passage_items:
                direct_answer, summary_candidates = extract_summary_answer(scope_passage_items)
                selected_items = scope_passage_items[: self.config.retrieval_top_k]
                final_passage_hash_ids = [x["hash_id"] for x in selected_items]
                final_passage_scores = self.build_scope_scores(selected_items, scope_score_breakdown)
                evidence_pages = sorted({x.get("page") for x in selected_items if isinstance(x.get("page"), int)})
                used_route = "scope_summary"
            elif target_doc_id and predicted_route == GENERAL_ROUTE and scope_passage_items:
                selected_items = scope_passage_items[: self.config.retrieval_top_k]
                final_passage_hash_ids = [x["hash_id"] for x in selected_items]
                final_passage_scores = self.build_scope_scores(selected_items, scope_score_breakdown)
                evidence_pages = sorted({x.get("page") for x in selected_items if isinstance(x.get("page"), int)})
                direct_answer = extract_presence_check_answer(question, scope_passage_items) or None
                used_route = "scope_general"
                if scope_image_items:
                    scope_retrieved_images = scope_image_items[: self.config.retrieval_top_k_image]
            elif target_doc_id:
                fallback_used = True
                sorted_passage_indices,sorted_passage_scores = self.dense_passage_retrieval(
                    question_embedding, allowed_passage_ids=allowed_passage_ids
                )
                if question_type in {"title", "summary"}:
                    page_candidates = self.collect_top_candidate_pages(sorted_passage_indices[: max(self.config.retrieval_top_k * 4, 10)])
                    logger.info("debug_structure title_or_summary target=%s top_page_candidates=%s", target_doc_id, page_candidates[:5])
                if question_type in {"first_figure_token", "quant_plus_first_figure", "chart_numeric"}:
                    anchor_pages = self.get_earliest_anchor_pages(target_doc_id)
                    logger.info("debug_structure first_figure target=%s earliest_anchor_pages=%s", target_doc_id, anchor_pages[:5])
                rerank_pool = max(self.config.retrieval_top_k * 4, self.config.retrieval_top_k)
                pool_indices = sorted_passage_indices[:rerank_pool]
                pool_hash_ids = [self.passage_embedding_store.hash_ids[idx] for idx in pool_indices]
                pool_scores = sorted_passage_scores[:rerank_pool]
                final_passage_hash_ids, final_passage_scores = self.rerank_passages_with_structure(
                    pool_hash_ids, pool_scores, question_type
                )
                final_passage_hash_ids = final_passage_hash_ids[:self.config.retrieval_top_k]
                final_passage_scores = final_passage_scores[:self.config.retrieval_top_k]
                used_route = "doc_aware_dense"
            elif len(seed_entities) != 0:
                fallback_used = True
                sorted_passage_hash_ids,sorted_passage_scores = self.graph_search_with_seed_entities(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
                final_passage_hash_ids = sorted_passage_hash_ids[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                used_route = "graph_seed"
            else:
                fallback_used = True
                sorted_passage_indices,sorted_passage_scores = self.dense_passage_retrieval(question_embedding)
                final_passage_indices = sorted_passage_indices[:self.config.retrieval_top_k]
                final_passage_hash_ids = [self.passage_embedding_store.hash_ids[idx] for idx in final_passage_indices]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                used_route = "dense_global"
            final_passages = [self.get_display_passage_text(hid) for hid in final_passage_hash_ids]

            retrieved_images = list(scope_retrieved_images)
            if self.config.use_image_retrieval and predicted_route == GENERAL_ROUTE:
                if retrieved_images:
                    pass
                elif target_doc_id and allowed_image_ids == []:
                    image_retrieval_skipped = True
                    image_retrieval_skipped_reason = "target_doc_has_no_indexed_images"
                    logger.info("image_retrieval skipped target_doc_id=%s reason=%s", target_doc_id, image_retrieval_skipped_reason)
                else:
                    image_indices, image_scores = self.dense_image_retrieval(question, allowed_image_ids=allowed_image_ids)
                    top_img_k = self.config.retrieval_top_k_image
                    for img_idx, img_score in zip(image_indices[:top_img_k], image_scores[:top_img_k]):
                        image_hash_id = self.image_embedding_store.hash_ids[img_idx]
                        image_meta = self.image_metadata_by_hash.get(image_hash_id, {})
                        image_doc_id = normalize_doc_id(image_meta.get("doc_id", ""))
                        if target_doc_id and image_doc_id != target_doc_id:
                            continue
                        retrieved_images.append({
                            "hash_id": image_hash_id,
                            "path": self.image_embedding_store.texts[img_idx],
                            "score": float(img_score),
                            "doc_id": image_doc_id,
                        })
                    final_passage_hash_ids, final_passages, final_passage_scores = self.fuse_passage_with_images(
                        final_passage_hash_ids, final_passages, final_passage_scores, retrieved_images
                    )

            if retrieved_images:
                retrieved_images = self.enrich_retrieved_images(retrieved_images)
            top_passage_doc_ids = [self.passage_metadata_by_hash.get(hid, {}).get("doc_id", "") for hid in final_passage_hash_ids]
            top_image_hash_ids = [item.get("hash_id", "") for item in retrieved_images[: self.config.retrieval_top_k_image]]
            top_image_doc_ids = [item.get("doc_id", "") for item in retrieved_images[: self.config.retrieval_top_k_image]]
            top_image_scores = [float(item.get("score", 0.0)) for item in retrieved_images[: self.config.retrieval_top_k_image]]
            image_passage_overlap = self.build_image_passage_overlap(retrieved_images, final_passage_hash_ids) if retrieved_images else []
            if target_doc_id and top_passage_doc_ids and any(doc_id != target_doc_id for doc_id in top_passage_doc_ids):
                logger.warning("doc_hint mismatch passages: target=%s, top_docs=%s", target_doc_id, top_passage_doc_ids)
            if target_doc_id and top_image_doc_ids and any(doc_id != target_doc_id for doc_id in top_image_doc_ids if doc_id):
                logger.warning("doc_hint mismatch images: target=%s, top_docs=%s", target_doc_id, top_image_doc_ids)

            result = {
                "question": raw_question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "retrieved_images": retrieved_images,
                "gold_answer": extract_gold_payload(question_info),
                "target_doc_id": target_doc_id,
                "question_type": question_type,
                "predicted_route": predicted_route,
                "route_source": route_source,
                "route_confidence": route_confidence,
                "top_route_scores": top_route_scores,
                "matched_prototypes": matched_prototypes,
                "evidence_plan": evidence_plan.to_dict(),
                "plan_slots": list(evidence_plan.evidence_slots),
                "plan_constraints": list(evidence_plan.structural_constraints),
                "candidate_count": candidate_count,
                "selected_evidence_ids": selected_evidence_ids,
                "selected_evidence_pages": evidence_pages,
                "selected_slot_assignments": selected_slot_assignments,
                "scope_score_breakdown": scope_score_breakdown,
                "structural_feature_hits": structural_feature_hits,
                "candidate_reliability": candidate_reliability,
                "slot_reliability": slot_reliability,
                "plan_reliability": plan_reliability,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "used_route": used_route,
                "seed_entities": seed_entities,
                "top_passage_doc_ids": top_passage_doc_ids,
                "top_image_hash_ids": top_image_hash_ids,
                "top_image_doc_ids": top_image_doc_ids,
                "top_image_scores": top_image_scores,
                "image_passage_overlap": image_passage_overlap,
                "evidence_pages": evidence_pages,
                "title_candidates": title_candidates,
                "figure_token_candidates": figure_token_candidates,
                "summary_candidates": summary_candidates,
                "quant_candidates": quant_candidates,
                "chart_numeric_candidates": chart_numeric_candidates,
                "chosen_quant_statement": chosen_quant_statement,
                "chosen_first_figure_token": chosen_first_figure_token,
                "image_retrieval_skipped": image_retrieval_skipped,
                "image_retrieval_skipped_reason": image_retrieval_skipped_reason,
                "parser_coverage_warning": parser_coverage_warning,
                "scope_selected_evidence": scope_selected_evidence,
                "direct_answer": direct_answer,
            }
            retrieval_results.append(result)
        return retrieval_results

    def get_display_passage_text(self, passage_hash_id):
        meta = self.passage_metadata_by_hash.get(passage_hash_id, {})
        text = meta.get("display_text") or self.passage_embedding_store.hash_id_to_text.get(passage_hash_id, "")
        doc_id = meta.get("doc_id")
        page = meta.get("page")
        if doc_id and page is not None:
            return f"[Doc: {doc_id} | Page: {page}] {text}"
        elif doc_id:
            return f"[Doc: {doc_id}] {text}"
        return text

    def collect_top_candidate_pages(self, passage_indices):
        page_count = defaultdict(int)
        for idx in passage_indices:
            if idx < 0 or idx >= len(self.passage_hash_ids):
                continue
            hid = self.passage_hash_ids[idx]
            page = self.passage_metadata_by_hash.get(hid, {}).get("page")
            if isinstance(page, int):
                page_count[page] += 1
        return sorted(page_count.items(), key=lambda x: (-x[1], x[0]))

    def build_scope_scores(self, selected_items, scope_score_breakdown):
        scores = []
        for item in selected_items:
            hid = item.get("hash_id")
            score = 0.0
            if hid:
                score = float(scope_score_breakdown.get(f"passage::{hid}", {}).get("final", 0.0))
            scores.append(score)
        return scores

    def run_scope_planner(self, question_text, target_doc_id, evidence_plan, question_info):
        if not target_doc_id:
            return [], [], {
                "fallback_used": True,
                "fallback_reason": "missing_target_doc",
                "candidate_count": 0,
                "plan_reliability": 0.0,
                "slot_reliability": {},
                "candidate_reliability": {},
                "structural_feature_hits": {},
            }
        candidates, doc_context = collect_plan_candidates(self, target_doc_id, evidence_plan, question_info)
        plan_result = plan_select_evidence(
            question_text,
            evidence_plan,
            candidates,
            doc_context,
            self.config.embedding_model,
        )
        selected_evidence = plan_result.get("selected_evidence", [])
        selected_passage_items = []
        selected_images = []
        seen_passages = set()
        for cand in selected_evidence:
            if cand["candidate_type"] == "image":
                image_hash = cand.get("metadata", {}).get("hash_id")
                if image_hash and image_hash in self.image_metadata_by_hash:
                    image_meta = dict(self.image_metadata_by_hash[image_hash])
                    image_meta["hash_id"] = image_hash
                    image_meta["score"] = float(plan_result.get("score_breakdown", {}).get(cand["candidate_id"], {}).get("final", 0.0))
                    selected_images.append(image_meta)
            for node_id in cand.get("source_node_ids", []):
                if node_id in self.passage_metadata_by_hash and node_id not in seen_passages:
                    row = dict(self.passage_metadata_by_hash[node_id])
                    row["hash_id"] = node_id
                    selected_passage_items.append(row)
                    seen_passages.add(node_id)
            for node_id in cand.get("metadata", {}).get("passage_hash_ids", []):
                if node_id in self.passage_metadata_by_hash and node_id not in seen_passages:
                    row = dict(self.passage_metadata_by_hash[node_id])
                    row["hash_id"] = node_id
                    selected_passage_items.append(row)
                    seen_passages.add(node_id)
        selected_passage_items = sorted(
            selected_passage_items,
            key=lambda x: (
                x.get("page") if isinstance(x.get("page"), int) else 10**9,
                int(x.get("chunk_idx", 10**6)),
            ),
        )
        return selected_passage_items, selected_images, {
            "candidate_count": len(candidates),
            "selected_evidence_ids": [x["candidate_id"] for x in selected_evidence],
            "selected_slot_assignments": plan_result.get("slot_assignments", {}),
            "scope_score_breakdown": plan_result.get("score_breakdown", {}),
            "structural_feature_hits": plan_result.get("planner_debug", {}).get("structural_feature_hits", {}),
            "candidate_reliability": plan_result.get("planner_debug", {}).get("candidate_reliability", {}),
            "slot_reliability": plan_result.get("planner_debug", {}).get("slot_reliability", {}),
            "plan_reliability": plan_result.get("planner_debug", {}).get("plan_reliability", 0.0),
            "fallback_reason": plan_result.get("planner_debug", {}).get("fallback_reason", ""),
            "selected_evidence": selected_evidence,
            "fallback_used": plan_result.get("fallback_used", False) or not bool(selected_passage_items or selected_images),
        }

    def get_earliest_anchor_pages(self, doc_id):
        doc_id = normalize_doc_id(doc_id)
        if not doc_id:
            return []
        pages = set()
        for meta in self.passage_metadata_by_hash.values():
            if normalize_doc_id(meta.get("doc_id", "")) != doc_id:
                continue
            anchors = meta.get("anchors") or []
            page = meta.get("page")
            text = meta.get("text_for_embed", "") or ""
            has_token_ref = bool(re.search(r"\b(?:fig(?:ure)?\.?|table)\s*\d+[A-Za-z]?\b", text, flags=re.IGNORECASE))
            if (anchors or has_token_ref) and isinstance(page, int):
                pages.add(page)
        return sorted(pages)

    def rerank_passages_with_structure(self, passage_hash_ids, passage_scores, question_type):
        if not passage_hash_ids:
            return [], []
        score_map = {}
        top_anchor = set(passage_hash_ids[:3])
        for hid, dense_score in zip(passage_hash_ids, passage_scores):
            meta = self.passage_metadata_by_hash.get(hid, {})
            score = float(dense_score)
            page = meta.get("page")
            anchors = meta.get("anchors") or []
            # Lightweight priors for structure-sensitive question styles.
            if question_type in {"title", "summary"} and isinstance(page, int):
                if page <= 2:
                    score += 0.08
                elif page <= 4:
                    score += 0.03
            if question_type in {"first_figure_token", "quant_plus_first_figure", "chart_numeric"} and anchors:
                score += 0.08
            # Preserve local continuity around very top dense passages.
            if hid not in top_anchor and self.is_adjacent_to_any(hid, top_anchor):
                score += 0.04
            score_map[hid] = score
        sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_items], [x[1] for x in sorted_items]

    def is_adjacent_to_any(self, passage_hash_id, other_passage_hash_ids):
        if not other_passage_hash_ids:
            return False
        neighbors = self.pdf_structure_state.get("passage_adjacent_map", {}).get(passage_hash_id, set())
        if not neighbors:
            return False
        return any(x in neighbors for x in other_passage_hash_ids)

    def fuse_passage_with_images(self, passage_hash_ids, passages, passage_scores, retrieved_images):
        if not passage_hash_ids or not retrieved_images:
            return passage_hash_ids, passages, passage_scores

        score_map = {hid: float(score) for hid, score in zip(passage_hash_ids, passage_scores)}
        # Boost passages when top retrieved image likely comes from the same PDF.
        for item in retrieved_images:
            image_score = item["score"]
            pdf_identifier = item.get("doc_id", "")
            if not pdf_identifier:
                continue
            for hid in passage_hash_ids:
                passage_doc_id = self.passage_metadata_by_hash.get(hid, {}).get("doc_id", "")
                if passage_doc_id == pdf_identifier:
                    score_map[hid] += self.config.image_ratio * image_score

        sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        sorted_hash_ids = [hid for hid, _ in sorted_items]
        sorted_scores = [score for _, score in sorted_items]
        sorted_passages = [self.get_display_passage_text(hid) for hid in sorted_hash_ids]
        return sorted_hash_ids, sorted_passages, sorted_scores

    def _precompute_sparse_matrices(self):
        """
        Precompute and cache sparse adjacency matrices for efficient vectorized retrieval using PyTorch.
        This is called once at the beginning of retrieve() to avoid rebuilding matrices per query.
        """
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # Build entity-to-sentence matrix (Mention matrix) using COO format
        entity_to_sentence_indices = []
        entity_to_sentence_values = []
        
        for entity_hash_id, sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.items():
            entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)
        
        # Build sentence-to-entity matrix
        sentence_to_entity_indices = []
        sentence_to_entity_values = []
        
        for sentence_hash_id, entity_hash_ids in self.sentence_hash_id_to_entity_hash_ids.items():
            sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)
        
        # Convert to PyTorch sparse tensors (COO format, then convert to CSR for efficiency)
        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                e2s_indices, e2s_values, (num_entities, num_sentences), device=self.device
            ).coalesce()
        else:
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences), device=self.device
            )
        
        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                s2e_indices, s2e_values, (num_sentences, num_entities), device=self.device
            ).coalesce()
        else:
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities), device=self.device
            )
            
    def graph_search_with_seed_entities(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        if self.config.use_vectorized_retrieval:
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        else:
            entity_weights, actived_entities = self.calculate_entity_scores(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        passage_weights = self.calculate_passage_scores(question_embedding,actived_entities)
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices,ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices,ppr_sorted_passage_scores

    def run_ppr(self, node_weights):        
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]
        
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]] 
            for i in sorted_indices_in_doc_scores
        ]
        
        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for seed_entity_idx,seed_entity,seed_entity_hash_id,seed_entity_score in zip(seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score    
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {}
            for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_hash_ids = [sid for sid in list(self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]) if sid not in used_sentence_hash_ids]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [self.sentence_embedding_store.hash_id_to_idx[sid] for sid in sentence_hash_ids]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][:self.config.top_k_sentence]
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        next_enitity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (next_enitity_node_idx, next_entity_score, iteration+1)
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, actived_entities

    def calculate_entity_scores_vectorized(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        """
        GPU-accelerated vectorized version using PyTorch sparse tensors.
        Uses sparse representation for both matrices and entity score vectors for maximum efficiency.
        Now includes proper dynamic pruning to match BFS behavior:
        - Sentence deduplication (tracks used sentences)
        - Per-entity top-k sentence selection
        - Proper threshold-based pruning
        """
        # Initialize entity weights
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # Compute all sentence similarities with the question at once
        question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
        sentence_similarities_np = np.dot(self.sentence_embeddings, question_emb).flatten()
        
        # Convert to torch tensors and move to device
        sentence_similarities = torch.from_numpy(sentence_similarities_np).float().to(self.device)
        
        # Track used sentences for deduplication (like BFS version)
        used_sentence_mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
        
        # Initialize seed entity scores as sparse tensor
        seed_indices = torch.tensor([[idx] for idx in seed_entity_indices], dtype=torch.long).t()
        seed_values = torch.tensor(seed_entity_scores, dtype=torch.float32)
        entity_scores_sparse = torch.sparse_coo_tensor(
            seed_indices, seed_values, (num_entities,), device=self.device
        ).coalesce()
        
        # Also maintain a dense accumulator for total scores
        entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
        entity_scores_dense.scatter_(0, torch.tensor(seed_entity_indices, device=self.device), 
                                     torch.tensor(seed_entity_scores, dtype=torch.float32, device=self.device))
        
        # Initialize actived_entities
        actived_entities = {}
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 0)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        
        current_entity_scores_sparse = entity_scores_sparse
        
        # Iterative matrix-based propagation using sparse matrices on GPU
        for iteration in range(1, self.config.max_iterations):
            # Convert sparse tensor to dense for threshold operation
            current_entity_scores_dense = current_entity_scores_sparse.to_dense()
            
            # Apply threshold to current scores
            current_entity_scores_dense = torch.where(
                current_entity_scores_dense >= self.config.iteration_threshold, 
                current_entity_scores_dense, 
                torch.zeros_like(current_entity_scores_dense)
            )
            
            # Get non-zero indices for sparse representation
            nonzero_mask = current_entity_scores_dense > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(nonzero_indices) == 0:
                break
            
            # Extract non-zero values and create sparse tensor
            nonzero_values = current_entity_scores_dense[nonzero_indices]
            current_entity_scores_sparse = torch.sparse_coo_tensor(
                nonzero_indices.unsqueeze(0), nonzero_values, (num_entities,), device=self.device
            ).coalesce()
            
            # Step 1: Sparse entity scores @ Sparse E2S matrix
            # Convert sparse vector to 2D for matrix multiplication
            current_scores_2d = torch.sparse_coo_tensor(
                torch.stack([nonzero_indices, torch.zeros_like(nonzero_indices)]),
                nonzero_values,
                (num_entities, 1),
                device=self.device
            ).coalesce()
            
            # E @ E2S -> sentence activation scores (sparse @ sparse = dense)
            sentence_activation = torch.sparse.mm(
                self.entity_to_sentence_sparse.t(),
                current_scores_2d
            )
            # Convert to dense before squeeze to avoid CUDA sparse tensor issues
            if sentence_activation.is_sparse:
                sentence_activation = sentence_activation.to_dense()
            sentence_activation = sentence_activation.squeeze()
            
            # Apply sentence deduplication: mask out used sentences
            sentence_activation = torch.where(
                used_sentence_mask,
                torch.zeros_like(sentence_activation),
                sentence_activation
            )
            
            # Step 2: Per-entity top-k sentence selection
            # This matches BFS behavior: each entity independently selects its top-k sentences
            selected_sentence_indices_list = []
            
            if len(nonzero_indices) > 0 and self.config.top_k_sentence > 0:
                # Iterate through each active entity
                for i, entity_idx in enumerate(nonzero_indices):
                    entity_score = nonzero_values[i]
                    
                    # Get sentences connected to this entity from the sparse matrix
                    # entity_to_sentence_sparse shape: (num_entities, num_sentences)
                    entity_row = self.entity_to_sentence_sparse[entity_idx].coalesce()
                    entity_sentence_indices = entity_row.indices()[0]  # Get column indices
                    
                    if len(entity_sentence_indices) == 0:
                        continue
                    
                    # Filter out already used sentences
                    sentence_mask = ~used_sentence_mask[entity_sentence_indices]
                    available_sentence_indices = entity_sentence_indices[sentence_mask]
                    
                    if len(available_sentence_indices) == 0:
                        continue
                    
                    # Get sentence similarities (for ranking)
                    sentence_sims = sentence_similarities[available_sentence_indices]
                    
                    # Select top-k sentences based ONLY on sentence similarity (matches BFS line 240)
                    # NOT weighted by entity_score at selection time
                    k = min(self.config.top_k_sentence, len(sentence_sims))
                    if k > 0:
                        top_k_values, top_k_local_indices = torch.topk(sentence_sims, k)
                        top_k_sentence_indices = available_sentence_indices[top_k_local_indices]
                        selected_sentence_indices_list.append(top_k_sentence_indices)
                
                # Merge all selected sentences (with deduplication via unique)
                if len(selected_sentence_indices_list) > 0:
                    all_selected_sentences = torch.cat(selected_sentence_indices_list)
                    unique_selected_sentences = torch.unique(all_selected_sentences)
                    
                    # Mark selected sentences as used
                    used_sentence_mask[unique_selected_sentences] = True
                    
                    # Compute weighted sentence scores for propagation
                    # weighted_score = sentence_activation * sentence_similarity
                    weighted_sentence_scores = sentence_activation * sentence_similarities
                    
                    # Zero out non-selected sentences
                    mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
                    mask[unique_selected_sentences] = True
                    weighted_sentence_scores = torch.where(
                        mask,
                        weighted_sentence_scores,
                        torch.zeros_like(weighted_sentence_scores)
                    )
                else:
                    # No sentences selected, create zero vector
                    weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            else:
                # No active entities or top_k_sentence is 0
                weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            
            # Step 3: Weighted sentences @ S2E -> propagate to next entities
            # Convert to sparse for more efficient computation
            weighted_nonzero_mask = weighted_sentence_scores > 0
            weighted_nonzero_indices = torch.nonzero(weighted_nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(weighted_nonzero_indices) > 0:
                weighted_nonzero_values = weighted_sentence_scores[weighted_nonzero_indices]
                weighted_scores_2d = torch.sparse_coo_tensor(
                    torch.stack([weighted_nonzero_indices, torch.zeros_like(weighted_nonzero_indices)]),
                    weighted_nonzero_values,
                    (num_sentences, 1),
                    device=self.device
                ).coalesce()
                
                next_entity_scores_result = torch.sparse.mm(
                    self.sentence_to_entity_sparse.t(),
                    weighted_scores_2d
                )
                # Convert to dense before squeeze to avoid CUDA sparse tensor issues
                if next_entity_scores_result.is_sparse:
                    next_entity_scores_result = next_entity_scores_result.to_dense()
                next_entity_scores_dense = next_entity_scores_result.squeeze()
            else:
                next_entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
            
            # Update entity scores (accumulate in dense format)
            entity_scores_dense += next_entity_scores_dense
            
            # Update actived_entities dictionary (record last trigger like BFS)
            # This matches BFS behavior: unconditionally update for entities above threshold
            next_entity_scores_np = next_entity_scores_dense.cpu().numpy()
            active_indices = np.where(next_entity_scores_np >= self.config.iteration_threshold)[0]
            for entity_idx in active_indices:
                score = next_entity_scores_np[entity_idx]
                entity_hash_id = self.entity_hash_ids[entity_idx]
                # Unconditionally update to record the last trigger (matches BFS line 252)
                actived_entities[entity_hash_id] = (entity_idx, float(score), iteration)
            
            # Prepare sparse tensor for next iteration
            next_nonzero_mask = next_entity_scores_dense > 0
            next_nonzero_indices = torch.nonzero(next_nonzero_mask, as_tuple=False).squeeze(-1)
            if len(next_nonzero_indices) > 0:
                next_nonzero_values = next_entity_scores_dense[next_nonzero_indices]
                current_entity_scores_sparse = torch.sparse_coo_tensor(
                    next_nonzero_indices.unsqueeze(0), next_nonzero_values, 
                    (num_entities,), device=self.device
                ).coalesce()
            else:
                break
        
        # Convert back to numpy for final processing
        entity_scores_final = entity_scores_dense.cpu().numpy()
        
        # Map entity scores to graph node weights (only for non-zero scores)
        nonzero_indices = np.where(entity_scores_final > 0)[0]
        for entity_idx in nonzero_indices:
            score = entity_scores_final[entity_idx]
            entity_hash_id = self.entity_hash_ids[entity_idx]
            entity_node_idx = self.node_name_to_vertex_idx[entity_hash_id]
            entity_weights[entity_node_idx] = float(score)
        
        return entity_weights, actived_entities

    def calculate_passage_scores(self, question_embedding, actived_entities):
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus
            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)
            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight
        return passage_weights

    def dense_passage_retrieval(self, question_embedding, allowed_passage_ids=None):
        question_emb = question_embedding.reshape(1, -1)
        if allowed_passage_ids:
            allowed_set = set(allowed_passage_ids)
            allowed_indices = [i for i, hid in enumerate(self.passage_hash_ids) if hid in allowed_set]
            if not allowed_indices:
                return np.array([], dtype=np.int64), []
            allowed_embs = self.passage_embeddings[allowed_indices]
            similarities = np.dot(allowed_embs, question_emb.T).flatten()
            local_sorted = np.argsort(similarities)[::-1]
            sorted_passage_indices = np.array([allowed_indices[i] for i in local_sorted], dtype=np.int64)
            sorted_passage_scores = similarities[local_sorted].tolist()
            return sorted_passage_indices, sorted_passage_scores

        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores

    def dense_image_retrieval(self, question, allowed_image_ids=None):
        if len(self.image_embeddings) == 0:
            return np.array([], dtype=np.int64), []
        question_image_embedding = self.vision_encoder.encode_text(question, normalize_embeddings=True)[0]
        question_image_embedding = question_image_embedding.reshape(1, -1)

        if allowed_image_ids is not None:
            allowed_set = set(allowed_image_ids)
            allowed_indices = [i for i, hid in enumerate(self.image_hash_ids) if hid in allowed_set]
            if not allowed_indices:
                return np.array([], dtype=np.int64), []
            allowed_embs = self.image_embeddings[allowed_indices]
            similarities = np.dot(allowed_embs, question_image_embedding.T).flatten()
            local_sorted = np.argsort(similarities)[::-1]
            sorted_image_indices = np.array([allowed_indices[i] for i in local_sorted], dtype=np.int64)
            sorted_image_scores = similarities[local_sorted].tolist()
            return sorted_image_indices, sorted_image_scores

        image_similarities = np.dot(self.image_embeddings, question_image_embedding.T).flatten()
        sorted_image_indices = np.argsort(image_similarities)[::-1]
        sorted_image_scores = image_similarities[sorted_image_indices].tolist()
        return sorted_image_indices, sorted_image_scores
    
    def get_seed_entities(self, question):
        question_entities = list(self.spacy_ner.question_ner(question))
        if len(question_entities) == 0:
            return [],[],[],[]
        question_entity_embeddings = self.config.embedding_model.encode(question_entities,normalize_embeddings=True,show_progress_bar=False,batch_size=self.config.batch_size)
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []       
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)
        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def index(self, passages, images_data, pdf_data):
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        self.passage_metadata_by_hash = {}
        self.image_metadata_by_hash = {}

        passage_texts_for_embed = []
        passage_item_ids = []
        for idx, item in enumerate(passages):
            if isinstance(item, dict) and item.get("passage_id"):
                passage_item_ids.append(item["passage_id"])
                passage_texts_for_embed.append(item.get("text_for_embed", item.get("display_text", "")))
            else:
                passage_item_ids.append(f"legacy:{idx}")
                passage_texts_for_embed.append(str(item))

        self.passage_embedding_store.insert_text(passage_texts_for_embed, item_ids=passage_item_ids)

        for idx, item in enumerate(passages):
            if isinstance(item, dict) and item.get("passage_id"):
                passage_id = item["passage_id"]
                passage_hash_id = compute_mdhash_id(str(passage_id), prefix="passage-")
                self.passage_metadata_by_hash[passage_hash_id] = {
                    "passage_id": passage_id,
                    "doc_id": normalize_doc_id(item.get("doc_id", "")),
                    "page": item.get("page"),
                    "chunk_idx": item.get("chunk_idx", idx),
                    "block_type": item.get("block_type", "text"),
                    "anchors": item.get("anchors", []),
                    "display_text": item.get("display_text", ""),
                    "text_for_embed": item.get("text_for_embed", ""),
                }
            else:
                passage_hash_id = compute_mdhash_id(f"legacy:{idx}", prefix="passage-")
                text = str(item)
                self.passage_metadata_by_hash[passage_hash_id] = {
                    "passage_id": f"legacy:{idx}",
                    "doc_id": "",
                    "page": None,
                    "chunk_idx": idx,
                    "block_type": "legacy",
                    "anchors": [],
                    "display_text": text,
                    "text_for_embed": text,
                }

        if self.config.use_image_retrieval and len(images_data) > 0:
            logger.info(f"正在索引 {len(images_data)} 张图像...")
            image_paths = [img["path"] for img in images_data]
            image_item_ids = [img.get("image_id", f"img:{i}") for i, img in enumerate(images_data)]
            self.image_embedding_store.insert_image(image_paths, item_ids=image_item_ids)
            for i, img in enumerate(images_data):
                image_hash_id = compute_mdhash_id(str(image_item_ids[i]), prefix="image-")
                self.image_metadata_by_hash[image_hash_id] = {
                    "image_id": image_item_ids[i],
                    "doc_id": normalize_doc_id(img.get("doc_id") or img.get("source_pdf") or ""),
                    "page": img.get("page"),
                    "path": img.get("path", ""),
                    "caption_hint": img.get("caption_hint"),
                    "source_pdf": img.get("source_pdf"),
                }

        self.doc_to_passage_hash_ids = build_doc_to_hash_ids(self.passage_metadata_by_hash)
        self.doc_to_image_hash_ids = build_doc_to_hash_ids(self.image_metadata_by_hash)
        self.pdf_structure_state = build_pdf_structure_state(
            pdf_data=pdf_data,
            passage_metadata_by_hash=self.passage_metadata_by_hash,
            image_metadata_by_hash=self.image_metadata_by_hash,
            infer_page_fn=infer_page_from_image_path,
        )
        self.image_to_passage_mapping = self.build_image_to_passage_mapping() if self.image_metadata_by_hash else {}
        os.makedirs(os.path.join(self.config.working_dir, self.dataset_name), exist_ok=True)
        with open(os.path.join(self.config.working_dir, self.dataset_name, "passage_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.passage_metadata_by_hash, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.config.working_dir, self.dataset_name, "image_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.image_metadata_by_hash, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.config.working_dir, self.dataset_name, "doc_to_passage_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(self.doc_to_passage_hash_ids, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.config.working_dir, self.dataset_name, "doc_to_image_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(self.doc_to_image_hash_ids, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.config.working_dir, self.dataset_name, "image_to_passage_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(self.image_to_passage_mapping, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.config.working_dir, self.dataset_name, "pdf_structure_state.json"), "w", encoding="utf-8") as f:
            json.dump(self._jsonable_structure_state(), f, ensure_ascii=False, indent=2)

        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        existing_passage_hash_id_to_entities,existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(hash_id_to_passage.keys())
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k : hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities,new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities)
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        entity_nodes, sentence_nodes,passage_hash_id_to_entities,self.entity_to_sentence,self.sentence_to_entity = self.extract_nodes_and_edges(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [self.sentence_embedding_store.text_to_hash_id[s] for s in sentence]
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [self.entity_embedding_store.text_to_hash_id[e] for e in entities]
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.add_pdf_structure_edges()
        self.augment_graph()
        output_graphml_path = os.path.join(self.config.working_dir,self.dataset_name, "LinearRAG.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)   
        self.graph.write_graphml(output_graphml_path)

    def add_adjacent_passage_edges(self):
        # Prefer explicit metadata order for PDF passages.
        doc_to_items = defaultdict(list)
        legacy_items = []
        for hid, meta in self.passage_metadata_by_hash.items():
            doc_id = meta.get("doc_id") or ""
            page = meta.get("page")
            chunk_idx = meta.get("chunk_idx", 0)
            if doc_id:
                page_order = page if isinstance(page, int) else 10**9
                doc_to_items[doc_id].append((page_order, int(chunk_idx), hid))
            else:
                legacy_items.append((int(chunk_idx), hid))

        for items in doc_to_items.values():
            items.sort(key=lambda x: (x[0], x[1]))
            for i in range(len(items) - 1):
                u = items[i][2]
                v = items[i + 1][2]
                self.node_to_node_stats[u][v] = max(self.node_to_node_stats[u].get(v, 0.0), 0.35)

        # Legacy fallback for non-PDF datasets.
        legacy_items.sort(key=lambda x: x[0])
        for i in range(len(legacy_items) - 1):
            u = legacy_items[i][1]
            v = legacy_items[i + 1][1]
            self.node_to_node_stats[u][v] = max(self.node_to_node_stats[u].get(v, 0.0), 1.0)

    def add_pdf_structure_edges(self):
        for u, v, edge_type, weight in self.pdf_structure_state.get("edges", []):
            if u == v:
                continue
            self.node_to_node_stats[u][v] = max(self.node_to_node_stats[u].get(v, 0.0), float(weight))

    def _jsonable_structure_state(self):
        out = dict(self.pdf_structure_state)
        if "passage_adjacent_map" in out:
            out["passage_adjacent_map"] = {
                k: sorted(list(v)) for k, v in out["passage_adjacent_map"].items()
            }
        return out

    def build_image_to_passage_mapping(self):
        if not self.config.use_image_retrieval:
            return {}
        page_nodes = self.pdf_structure_state.get("page_nodes", {})
        mapping = {}
        for page_node_id, page_meta in page_nodes.items():
            doc_id = normalize_doc_id(page_meta.get("doc_id", ""))
            page = page_meta.get("page")
            related_passage_ids = sorted(page_meta.get("passage_hash_ids", []))
            for image_hash_id in page_meta.get("image_hash_ids", []):
                if image_hash_id in mapping:
                    raise ValueError(f"Duplicate page mapping detected for image {image_hash_id}: {mapping[image_hash_id]['page_node_id']} and {page_node_id}")
                image_meta = self.image_metadata_by_hash.get(image_hash_id)
                if image_meta is None:
                    raise KeyError(f"Image {image_hash_id} is linked from structure state but missing from image metadata.")
                image_doc_id = normalize_doc_id(image_meta.get("doc_id", ""))
                image_page = image_meta.get("page")
                if image_doc_id != doc_id:
                    raise ValueError(f"Image {image_hash_id} doc mismatch: metadata={image_doc_id}, structure={doc_id}")
                if isinstance(image_page, int) and image_page != page:
                    raise ValueError(f"Image {image_hash_id} page mismatch: metadata={image_page}, structure={page}")
                mapping[image_hash_id] = {
                    "image_hash_id": image_hash_id,
                    "doc_id": doc_id,
                    "page": page,
                    "page_node_id": page_node_id,
                    "related_passage_ids": related_passage_ids,
                }
        missing_image_mappings = sorted(set(self.image_metadata_by_hash.keys()) - set(mapping.keys()))
        if missing_image_mappings:
            raise KeyError(
                "Missing page/passage mapping for image hash ids: " + ", ".join(missing_image_mappings[:10])
            )
        return mapping

    def require_image_mapping(self, image_hash_id):
        mapping = self.image_to_passage_mapping.get(image_hash_id)
        if mapping is None:
            raise KeyError(f"Missing explicit image-to-passage mapping for image hash id: {image_hash_id}")
        return mapping

    def enrich_retrieved_images(self, retrieved_images):
        enriched = []
        for item in retrieved_images:
            image_hash_id = item.get("hash_id")
            if not image_hash_id:
                raise ValueError(f"Retrieved image item is missing hash_id: {item}")
            mapping = self.require_image_mapping(image_hash_id)
            merged = dict(item)
            merged.setdefault("doc_id", mapping.get("doc_id", ""))
            merged.setdefault("page", mapping.get("page"))
            merged["related_passage_ids"] = list(mapping.get("related_passage_ids", []))
            enriched.append(merged)
        return enriched

    def build_image_passage_overlap(self, retrieved_images, selected_passage_hash_ids):
        selected_set = set(selected_passage_hash_ids)
        overlap_rows = []
        for item in retrieved_images[: self.config.retrieval_top_k_image]:
            image_hash_id = item.get("hash_id")
            mapping = self.require_image_mapping(image_hash_id)
            related_passage_ids = list(mapping.get("related_passage_ids", []))
            overlap_ids = [hid for hid in related_passage_ids if hid in selected_set]
            overlap_rows.append({
                "image_hash_id": image_hash_id,
                "doc_id": mapping.get("doc_id", ""),
                "page": mapping.get("page"),
                "related_passage_ids": related_passage_ids,
                "selected_passage_overlap": overlap_ids,
                "overlap_count": len(overlap_ids),
            })
        return overlap_rows

    def augment_graph(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()} 
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        image_hash_id_to_text = {}
        if self.config.use_image_retrieval and hasattr(self, "image_embedding_store"):
            image_hash_id_to_text = self.image_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}
        
        passage_hash_ids = set(passage_hash_id_to_text.keys())
        image_hash_ids = set(image_hash_id_to_text.keys())
        
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                node_type = "passage" if hash_id in passage_hash_ids else "entity"
                self.graph.add_vertex(name=hash_id, content=text, node_type=node_type)
        for hash_id, text in image_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                image_meta = self.image_metadata_by_hash.get(hash_id, {})
                self.graph.add_vertex(
                    name=hash_id,
                    content=text,
                    node_type="image",
                    doc_id=image_meta.get("doc_id", ""),
                    page=image_meta.get("page"),
                )

        # Add explicit structure nodes (page / visual).
        for page_node_id, page_meta in self.pdf_structure_state.get("page_nodes", {}).items():
            if page_node_id in existing_nodes:
                continue
            self.graph.add_vertex(
                name=page_node_id,
                content=f"[PAGE] {page_meta.get('doc_id', '')} p{page_meta.get('page', '')}",
                node_type="page",
                doc_id=page_meta.get("doc_id", ""),
                page=page_meta.get("page"),
                is_front_matter=bool(page_meta.get("is_front_matter", False)),
            )
        for visual_node_id, visual_meta in self.pdf_structure_state.get("visual_nodes", {}).items():
            if visual_node_id in existing_nodes:
                continue
            self.graph.add_vertex(
                name=visual_node_id,
                content=f"[VISUAL_REF] {visual_meta.get('anchor', '')}",
                node_type="visual",
                doc_id=visual_meta.get("doc_id", ""),
                anchor=visual_meta.get("anchor", ""),
            )
        
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}   
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id] 
            for passage_id in passage_hash_ids 
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self):
        edge_weight_map = {}
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                u, v = sorted([node_hash_id, neighbor_hash_id])
                edge_weight_map[(u, v)] = max(edge_weight_map.get((u, v), 0.0), float(weight))
        edges = list(edge_weight_map.keys())
        weights = [edge_weight_map[e] for e in edges]
        if not edges:
            return
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        passage_to_entity_count ={} 
        passage_to_all_score = defaultdict(int)
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity)
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            score = count / passage_to_all_score[passage_hash_id]
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence= defaultdict(set)
        sentence_to_entity = defaultdict(set)
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        for sentence,entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        with open(self.ner_results_path, "w") as f:
            json.dump({"passage_hash_id_to_entities": existing_passage_hash_id_to_entities, "sentence_to_entities": existing_sentence_to_entities}, f)

    def inspect_doc_structure(self, doc_id, max_pages=5):
        doc_id = normalize_doc_id(doc_id)
        if not doc_id:
            return {"doc_id": "", "pages": [], "passage_links": [], "visual_links": []}
        page_nodes = self.pdf_structure_state.get("doc_to_page_nodes", {}).get(doc_id, [])
        page_nodes = page_nodes[:max_pages]
        page_meta = self.pdf_structure_state.get("page_nodes", {})
        out_pages = [page_meta.get(pid, {}) for pid in page_nodes]
        passage_links = []
        visual_links = []
        for u, v, etype, weight in self.pdf_structure_state.get("edges", []):
            if etype == "passage_page" and v in page_nodes:
                passage_links.append({"passage": u, "page_node": v, "weight": weight})
            if etype in {"passage_visual", "visual_page"}:
                v_doc = ""
                if v.startswith("visual::"):
                    v_doc = v.split("::", 2)[1]
                elif u.startswith("visual::"):
                    v_doc = u.split("::", 2)[1]
                if v_doc == doc_id:
                    visual_links.append({"src": u, "dst": v, "edge_type": etype, "weight": weight})
        return {
            "doc_id": doc_id,
            "pages": out_pages,
            "passage_links": passage_links[:30],
            "visual_links": visual_links[:30],
        }

    def inspect_question_structure(self, question, max_pages=5):
        target_doc_id, cleaned_question = parse_doc_id_from_question(question or "")
        return {
            "question": cleaned_question,
            "target_doc_id": target_doc_id,
            "question_type": infer_question_type(question or ""),
            "doc_structure": self.inspect_doc_structure(target_doc_id, max_pages=max_pages),
        }
