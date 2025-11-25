import os
import glob
import json
import re
import math
import faiss
import numpy as np
import pdfplumber
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

base_dir = "F:\\Freelancing\\Quiz"

domains = {
    "Science": os.path.join(base_dir, "Science"),
    "Computer Science": os.path.join(base_dir, "Computer Science")
}

embed_model_name = "sentence-transformers/all-mpnet-base-v2"
embedder = SentenceTransformer(embed_model_name)

def is_heading(line):
    if not line:
        return False
    s = line.strip()
    if len(s) < 4:
        return False
    words = s.split()
    if len(words) > 12:
        return False
    if s.isupper():
        return True
    if re.match(r'^[A-Z][\w\-]*(\s+[A-Z][\w\-]*){0,6}$', s):
        return True
    if s.endswith(":") and len(words) <= 8:
        return True
    return False

def split_into_paragraphs(text):
    if not text:
        return []
    paragraphs = []
    raw = text.splitlines()
    buf = []
    for line in raw:
        if line.strip() == "":
            if buf:
                paragraphs.append(" ".join(buf).strip())
                buf = []
            continue
        buf.append(line.strip())
    if buf:
        paragraphs.append(" ".join(buf).strip())
    return paragraphs

def make_chunks(paragraphs, size_words=320, overlap_words=80):
    out = []
    i = 0
    words_so_far = []
    while i < len(paragraphs):
        para_words = paragraphs[i].split()
        if len(words_so_far) + len(para_words) <= size_words or not words_so_far:
            words_so_far.extend(para_words)
            i += 1
            continue
        out.append(" ".join(words_so_far))
        if overlap_words > 0:
            words_so_far = words_so_far[-overlap_words:]
        else:
            words_so_far = []
    if words_so_far:
        out.append(" ".join(words_so_far))
    return out

def process_domain(domain_name, pdf_folder):
    logging.info(f"--- Starting ingestion for {domain_name} ---")
    
    pdf_paths = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")))
    logging.info(f"Found {len(pdf_paths)} PDFs in folder: {pdf_folder}")

    chunks = []
    chunk_id = 0
    chapter_map = {}

    for pdf_path in pdf_paths:
        logging.info(f"Processing {pdf_path}")
        current_chapter = f"{os.path.basename(pdf_path)} - Unknown Chapter"
        chapter_map.setdefault(current_chapter, [])
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                lines = page_text.splitlines()
                
                for line in lines:
                    if is_heading(line):
                        current_chapter = f"{os.path.basename(pdf_path)} - {line.strip()}"
                        chapter_map.setdefault(current_chapter, [])
                
                paragraphs = split_into_paragraphs(page_text)
                if not paragraphs:
                    continue
                
                page_chunks = make_chunks(paragraphs, size_words=320, overlap_words=80)
                for pc in page_chunks:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "book": os.path.basename(pdf_path),
                        "pdf_path": pdf_path,
                        "page": page_num,
                        "chapter": current_chapter,
                        "text": pc,
                        "word_count": len(pc.split())
                    })
                    chapter_map[current_chapter].append(chunk_id)
                    chunk_id += 1

    logging.info(f"Total chunks for {domain_name}: {len(chunks)}")

    if not chunks:
        logging.warning(f"No chunks created for {domain_name}. Skipping embedding and saving.")
        return

    texts = [c["text"] for c in chunks]
    batch_size = 64
    embeddings = np.zeros((len(texts), embedder.get_sentence_embedding_dimension()), dtype="float32")

    for i in range(0, len(texts), batch_size):
        logging.info(f"Embedding batch {i} to {min(i+batch_size, len(texts))} for {domain_name}")
        batch_texts = texts[i:i+batch_size]
        emb = embedder.encode(batch_texts, convert_to_numpy=True)
        embeddings[i:i+len(emb)] = emb.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    output_folder = domain_name.replace(" ", "_")
    os.makedirs(output_folder, exist_ok=True)

    faiss_path = os.path.join(output_folder, "faiss.index")
    metadata_path = os.path.join(output_folder, "metadata.json")
    chapter_map_path = os.path.join(output_folder, "chapter_map.json")

    logging.info(f"Writing FAISS index to {faiss_path}")
    faiss.write_index(index, faiss_path)

    logging.info(f"Saving metadata.json to {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logging.info(f"Saving chapter_map.json to {chapter_map_path}")
    with open(chapter_map_path, "w", encoding="utf-8") as f:
        json.dump(chapter_map, f, ensure_ascii=False, indent=2)

    logging.info(f"Ingestion for {domain_name} completed successfully")

for name, folder in domains.items():
    process_domain(name, folder)

logging.info("--- All ingestion processes completed ---")