from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


_semantic_model = None


def _get_semantic_model():
    """Lazy load SentenceTransformer model for semantic chunking."""
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    if not text or not text.strip():
        return []

    from numpy import dot
    from numpy.linalg import norm

    # Split text into sentences by punctuation or multiple newlines
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return []

    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic"})]

    model = _get_semantic_model()
    embeddings = model.encode(sentences)

    def _cosine_sim(a, b) -> float:
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    chunks: list[Chunk] = []
    current_sentences = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            joined_text = " ".join(current_sentences).strip()
            if joined_text:
                chunks.append(Chunk(
                    text=joined_text,
                    metadata={**metadata, "strategy": "semantic"}
                ))
            current_sentences = [sentences[i]]
        else:
            current_sentences.append(sentences[i])

    if current_sentences:
        joined_text = " ".join(current_sentences).strip()
        if joined_text:
            chunks.append(Chunk(
                text=joined_text,
                metadata={**metadata, "strategy": "semantic"}
            ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if not text or not text.strip():
        return ([], [])

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    parents: list[Chunk] = []
    children: list[Chunk] = []

    current_parent_paras: list[str] = []
    current_parent_len = 0
    parent_groups: list[str] = []

    for para in paragraphs:
        para_len = len(para)
        added_len = para_len + (2 if current_parent_paras else 0)
        if current_parent_paras and (current_parent_len + added_len > parent_size):
            parent_groups.append("\n\n".join(current_parent_paras))
            current_parent_paras = [para]
            current_parent_len = para_len
        else:
            current_parent_paras.append(para)
            current_parent_len += added_len

    if current_parent_paras:
        parent_groups.append("\n\n".join(current_parent_paras))

    for idx, parent_text in enumerate(parent_groups):
        pid = f"parent_{idx}"
        parent_chunk = Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            parent_id=pid,
        )
        parents.append(parent_chunk)

        sub_units = [u.strip() for u in re.split(r'(?<=[.!?])\s+|\n+', parent_text) if u.strip()]
        if not sub_units:
            sub_units = [parent_text]

        current_child_units: list[str] = []
        current_child_len = 0

        for unit in sub_units:
            if len(unit) > child_size:
                if current_child_units:
                    child_text = " ".join(current_child_units).strip()
                    if child_text:
                        children.append(Chunk(
                            text=child_text,
                            metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                            parent_id=pid,
                        ))
                    current_child_units = []
                    current_child_len = 0

                words = unit.split()
                temp_words: list[str] = []
                temp_len = 0
                for w in words:
                    w_add = len(w) + (1 if temp_words else 0)
                    if temp_words and (temp_len + w_add > child_size):
                        children.append(Chunk(
                            text=" ".join(temp_words).strip(),
                            metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                            parent_id=pid,
                        ))
                        temp_words = [w]
                        temp_len = len(w)
                    else:
                        temp_words.append(w)
                        temp_len += w_add
                if temp_words:
                    current_child_units = temp_words
                    current_child_len = temp_len
            else:
                unit_add = len(unit) + (1 if current_child_units else 0)
                if current_child_units and (current_child_len + unit_add > child_size):
                    child_text = " ".join(current_child_units).strip()
                    if child_text:
                        children.append(Chunk(
                            text=child_text,
                            metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                            parent_id=pid,
                        ))
                    current_child_units = [unit]
                    current_child_len = len(unit)
                else:
                    current_child_units.append(unit)
                    current_child_len += unit_add

        if current_child_units:
            child_text = " ".join(current_child_units).strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                    parent_id=pid,
                ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_header = ""
    current_lines: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block

        is_header = False
        if not in_code_block and re.match(r"^#{1,3}\s+", line):
            is_header = True

        if is_header:
            if current_header or current_lines:
                content = "\n".join(current_lines).strip()
                if current_header or content:
                    sections.append((current_header, content))
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_header or current_lines:
        content = "\n".join(current_lines).strip()
        if current_header or content:
            sections.append((current_header, content))

    chunks: list[Chunk] = []
    doc_title = ""

    for header, content in sections:
        clean_heading = header.lstrip("#").strip() if header else ""
        if header.startswith("# ") and not content:
            doc_title = clean_heading
            continue

        if header and content:
            chunk_text = f"{header}\n\n{content}".strip()
        elif header:
            chunk_text = header.strip()
        else:
            chunk_text = content.strip()

        if not chunk_text:
            continue

        section_name = clean_heading or doc_title or metadata.get("source", "General")
        meta = {
            **metadata,
            "section": section_name,
            "strategy": "structure",
        }
        chunks.append(Chunk(text=chunk_text, metadata=meta))

    if not chunks and text.strip():
        chunks.append(Chunk(
            text=text.strip(),
            metadata={**metadata, "section": metadata.get("source", "General"), "strategy": "structure"}
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
