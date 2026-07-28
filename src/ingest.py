"""
ingest.py
Loads Etisalat Egypt knowledge-base .docx files, adds the metadata the
project spec asks for (category, keywords, last-updated date), chunks
each document by heading section, embeds the chunks, and stores them
in a persistent ChromaDB collection.

Usage:
    python src/ingest.py
"""

import os
import re
import json
import glob
from datetime import date
from pathlib import Path

import docx
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "etisalat_kb"

# Map source filenames -> category + keywords.
# This is the metadata layer the project brief (section 4.2) asks for.
# Add an entry here whenever a new source document is added.
DOC_METADATA = {
    "DataLine": {
        "category": "Mobile Data Plans",
        "keywords": ["data line", "MiFi", "USB modem", "خط الداتا", "إنترنت"],
    },
    "PrepaidSystems": {
        "category": "Prepaid Plans",
        "keywords": ["prepaid", "أقوى كارت", "أهلا حسب الاستخدام", "منيو الجنيه"],
    },
    "Emerald": {
        "category": "Postpaid Plans",
        "keywords": ["postpaid", "اميرالد", "family plan", "roaming"],
    },
    "HekayaMixat": {
        "category": "Prepaid Plans",
        "keywords": ["حكاية ميكسات", "mix", "prepaid"],
    },
    "HekayaInternet": {
        "category": "Prepaid Plans - Internet Focused",
        "keywords": ["حكاية انترنت", "mobile internet", "prepaid"],
    },
}

TODAY = date.today().isoformat()


def clean_filename_key(filename: str) -> str:
    """Turn 'Emerald (2).docx' -> 'Emerald' so it matches DOC_METADATA."""
    stem = Path(filename).stem
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem).strip()
    return stem


def extract_sections(docx_path: Path):
    """
    Walk the paragraphs of a docx file and group them into sections
    based on Heading 1/2/3 styles. Returns a list of dicts:
        {heading_path, level, text}
    heading_path keeps parent headings so a chunk retains context,
    e.g. "2. الباقات الأساسية > الباقات بصلاحية شهر واحد".
    """
    document = docx.Document(str(docx_path))
    sections = []
    current = {"heading_path": [], "level": 0, "text": []}
    heading_stack = []

    def flush():
        if current["text"]:
            sections.append(
                {
                    "heading_path": " > ".join(heading_stack) if heading_stack else "Introduction",
                    "text": "\n".join(current["text"]).strip(),
                }
            )
        current["text"] = []

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style else "Normal"

        if style.startswith("Heading"):
            level = int(style.replace("Heading", "").strip() or 1)
            flush()
            # trim stack to this heading's level, then push
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(text)
        else:
            current["text"].append(text)

    flush()
    return [s for s in sections if s["text"]]


def chunk_sections(sections, max_words=220):
    """
    Merge/split section text into chunks of roughly max_words.
    Small consecutive sections under the same heading path get merged;
    long sections get split on paragraph boundaries.
    """
    chunks = []
    buffer_text = ""
    buffer_heading = None

    def flush_buffer():
        if buffer_text.strip():
            chunks.append({"heading_path": buffer_heading, "text": buffer_text.strip()})

    for sec in sections:
        words = sec["text"].split()
        if len(words) > max_words:
            flush_buffer()
            buffer_text, buffer_heading = "", None
            # split long section into paragraph-sized pieces
            paras = sec["text"].split("\n")
            piece = ""
            for p in paras:
                if len((piece + " " + p).split()) > max_words and piece:
                    chunks.append({"heading_path": sec["heading_path"], "text": piece.strip()})
                    piece = p
                else:
                    piece = (piece + "\n" + p).strip()
            if piece:
                chunks.append({"heading_path": sec["heading_path"], "text": piece.strip()})
            continue

        if buffer_heading == sec["heading_path"] or buffer_heading is None:
            candidate = (buffer_text + "\n" + sec["text"]).strip()
            if len(candidate.split()) <= max_words:
                buffer_text = candidate
                buffer_heading = sec["heading_path"]
                continue

        flush_buffer()
        buffer_text, buffer_heading = sec["text"], sec["heading_path"]

    flush_buffer()
    return chunks


def build_records():
    records = []
    files = sorted(glob.glob(str(RAW_DIR / "*.docx")))
    if not files:
        raise FileNotFoundError(f"No .docx files found in {RAW_DIR}")

    for f in files:
        f = Path(f)
        key = clean_filename_key(f.name)
        meta = DOC_METADATA.get(key, {"category": "General", "keywords": []})

        document = docx.Document(str(f))
        title = next((p.text.strip() for p in document.paragraphs
                      if p.style.name == "Heading 1" and p.text.strip()), key)

        sections = extract_sections(f)
        chunks = chunk_sections(sections)

        for i, ch in enumerate(chunks):
            records.append(
                {
                    "id": f"{key}-{i}",
                    "text": f"{title}\n{ch['heading_path']}\n{ch['text']}",
                    "metadata": {
                        "source": f.name,
                        "title": title,
                        "category": meta["category"],
                        "keywords": ", ".join(meta["keywords"]),
                        "heading_path": ch["heading_path"],
                        "chunk_index": i,
                        "last_updated": TODAY,
                    },
                }
            )

    return records


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()

    # Save a human-readable copy of the processed chunks for review/debugging
    with open(PROCESSED_DIR / "chunks.json", "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    print(f"Extracted {len(records)} chunks -> {PROCESSED_DIR / 'chunks.json'}")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")  # optional, for OpenAI-compatible providers
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. Skipping embedding step.")
        print("Set it in a .env file, then re-run this script to populate ChromaDB.")
        return

    embed_kwargs = {"api_key": api_key, "model_name": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")}
    if base_url:
        embed_kwargs["api_base"] = base_url
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(**embed_kwargs)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Fresh collection each run so re-ingesting doesn't duplicate.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=openai_ef)

    ids = [r["id"] for r in records]
    docs = [r["text"] for r in records]
    metas = [r["metadata"] for r in records]

    batch = 100
    for i in range(0, len(records), batch):
        collection.add(
            ids=ids[i : i + batch],
            documents=docs[i : i + batch],
            metadatas=metas[i : i + batch],
        )
        print(f"  embedded {min(i + batch, len(records))}/{len(records)}")

    print(f"Done. Collection '{COLLECTION_NAME}' stored at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
