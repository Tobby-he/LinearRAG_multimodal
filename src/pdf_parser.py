import json
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _pick_markdown_file(output_dir: Path, pdf_path: str) -> Path | None:
    stem = Path(pdf_path).stem
    candidates = list(output_dir.rglob("*.md"))
    if not candidates:
        return None

    # Prefer exact stem match first.
    for c in candidates:
        if c.stem == stem:
            return c

    # Fallback to the largest markdown file.
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def _collect_images(output_dir: Path, source_pdf: str) -> list:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = []
    for img_path in output_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in exts:
            images.append({"path": str(img_path.resolve()), "source_pdf": source_pdf})
    return images


def _run_mineru(pdf_path: str, output_dir: Path):
    # MinerU v2 command
    cmd_mineru = ["mineru", "-p", pdf_path, "-o", str(output_dir)]
    # Some environments still expose legacy command name.
    cmd_magic_pdf = ["magic-pdf", "-p", pdf_path, "-o", str(output_dir), "-m", "auto"]

    for cmd in (cmd_mineru, cmd_magic_pdf):
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("PDF parsed by command: %s", " ".join(cmd))
                return
            logger.warning("Command failed (%s): %s", " ".join(cmd), result.stderr[-500:])
        except FileNotFoundError:
            logger.warning("Command not found: %s", cmd[0])

    raise RuntimeError(
        "MinerU command not available. Please install MinerU and ensure `mineru` (or `magic-pdf`) is in PATH."
    )


def parse_pdf_to_text(pdf_path: str) -> tuple:
    """
    Parse a PDF with local MinerU and return:
    (full_text, extracted_images)
    """
    base_name = Path(pdf_path).name
    ts = int(time.time() * 1000)
    output_dir = Path("import") / "pdf_parse" / f"{base_name}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _run_mineru(pdf_path, output_dir)
        md_file = _pick_markdown_file(output_dir, pdf_path)
        full_text = md_file.read_text(encoding="utf-8", errors="ignore") if md_file else ""
        extracted_images = _collect_images(output_dir, pdf_path)
        logger.info("Parsed PDF '%s': markdown=%s, images=%d", pdf_path, str(md_file) if md_file else "None", len(extracted_images))
        return full_text, extracted_images
    except Exception as e:
        logger.error("Error parsing PDF '%s' with MinerU: %s", pdf_path, e)
        return "", []


def load_pdf_documents(pdf_paths_file: str) -> tuple:
    """
    Load PDF paths from JSON and parse existing files.
    Returns: (pdf_docs_data, all_pdf_images)
    """
    pdf_docs_data = []
    all_pdf_images = []

    if not pdf_paths_file:
        return pdf_docs_data, all_pdf_images
    if not os.path.exists(pdf_paths_file):
        logger.warning("PDF paths file not found: %s", pdf_paths_file)
        return pdf_docs_data, all_pdf_images

    with open(pdf_paths_file, "r", encoding="utf-8") as f:
        pdf_paths = json.load(f)

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    base_dir = os.path.dirname(os.path.abspath(pdf_paths_file))
    for item in pdf_paths:
        pdf_path = item.get("path", "") if isinstance(item, dict) else item
        if not pdf_path:
            continue
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(base_dir, pdf_path)
        if os.path.exists(pdf_path):
            pdf_content, pdf_images = parse_pdf_to_text(pdf_path)
            if pdf_content:
                pdf_docs_data.append({"path": pdf_path, "content": pdf_content})
                all_pdf_images.extend(pdf_images)
        else:
            logger.warning("PDF file not found: %s", pdf_path)

    logger.info("Loaded %d PDF docs and %d images.", len(pdf_docs_data), len(all_pdf_images))
    return pdf_docs_data, all_pdf_images
