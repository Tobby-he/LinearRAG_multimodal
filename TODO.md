# LinearRAG Multimodal TODO

> Goal: evolve current PDF+image indexing into a robust multimodal RAG pipeline.

## P0 - Must Do First

- [ ] Remove hard-coded debug truncation in `run.py`
  - Replace fixed `questions[:5]` and `passages[:100]` with CLI flags:
    - `--debug_limit_questions` (default: `-1` means disabled)
    - `--debug_limit_passages` (default: `-1` means disabled)

- [ ] Make multimodal retrieval deterministic and configurable
  - Add CLI/config args:
    - `--retrieval_top_k_image`
    - `--image_ratio`
  - Plumb them from `run.py` -> `LinearRAGConfig` -> retrieval code.

- [ ] Fix text/image fusion strategy
  - Current fusion boosts passages only by PDF filename match.
  - Upgrade to explicit mapping at indexing time:
    - `image_path -> source_pdf -> related passage hash_ids`
  - Persist mapping JSON under `import/<dataset>/`.

- [ ] Ensure passage IDs and metadata are explicit
  - When adding PDF chunks, store structured metadata:
    - source file, chunk id, optional page range
  - Keep raw text clean (avoid stuffing metadata into text body).

## P1 - Retrieval Quality

- [ ] Add image retrieval diagnostics
  - Save per-question top-k images and scores to predictions.
  - Save a small debug report: overlap between retrieved PDF chunks and image sources.

- [ ] Introduce rank-fusion baseline
  - Implement RRF or weighted normalized-score fusion:
    - text rank list + image-informed rank list.
  - Keep current heuristic as fallback via config switch.

- [ ] Add OCR/table handling policy for PDFs
  - If MinerU output has markdown tables/figures, normalize before chunking.
  - Add options for chunking by heading/page vs fixed length.

## P1 - Generator Side (True Multimodal)

- [ ] Upgrade `LLM_Model` to support multimodal chat payloads
  - Build message format that can include image inputs (path -> base64/data URL).
  - Keep text-only path backward compatible.

- [ ] In `qa()`, pass retrieved image evidence to model as images (not just path strings)
  - Add max image count and size constraints.
  - Add graceful fallback when model endpoint is text-only.

## P2 - Engineering Hygiene

- [ ] Add schema checks for input files
  - Validate `pdf_paths.json` / `image_paths.json` structure with clear errors.

- [ ] Add cache/versioning controls
  - Add `--rebuild_index` to force re-embed/re-parse.
  - Track index build fingerprint (model + chunk params + parser params).

- [ ] Add tests
  - Unit tests:
    - PDF path resolution
    - chunking behavior
    - image retrieval scoring
    - fusion ordering
  - Smoke test:
    - one tiny PDF + one question end-to-end.

- [ ] Improve logging and observability
  - Log counts at each stage:
    - PDF docs, chunks, images indexed, image hits used in fusion.

## P3 - Nice to Have

- [ ] Add reranker over final contexts (text + image captions)
- [ ] Add optional captioning for images to improve text-side reasoning
- [ ] Export retrieval traces for offline error analysis

## Next Session Suggested Start

1. Implement CLI debug-limit flags and remove hard-coded truncation.
2. Add configurable `retrieval_top_k_image` and `image_ratio` wiring.
3. Replace filename-based fusion with explicit image->passage mapping persisted in index.

## Acceptance Criteria

- [ ] No hard-coded debug truncation in production path.
- [ ] Running with `--use_pdf_retrieval` returns both `sorted_passage` and `retrieved_images` stably.
- [ ] Fusion behavior is configurable and reproducible.
- [ ] Multimodal payload path exists for supported LLM endpoints.
