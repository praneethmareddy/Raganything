# =========================
# UNIVERSAL MULTIMODAL RAG
# Single-cell implementation
# =========================

# -------- Imports --------
import os
import subprocess
import numpy as np
import pandas as pd
import faiss
import torch
import clip
import networkx as nx

from PIL import Image
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter
from unstructured.partition.auto import partition


# -------- Device --------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -------- Load Models --------
clip_model, preprocess = clip.load("ViT-L/14", device=device)
text_model = SentenceTransformer("BAAI/bge-base-en-v1.5")


# -------- Ingestion --------
def ingest(file_path):
    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        doc = DocumentConverter().convert(file_path)
        return {
            "type": "pdf",
            "text_blocks": doc.text_blocks,
            "tables": doc.tables,
            "figures": doc.figures,
            "formulas": getattr(doc, "formulas", [])
        }

    if ext in ["docx", "pptx"]:
        elements = partition(file_path)
        texts = [e.text for e in elements if hasattr(e, "text") and e.text]
        return {"type": "doc", "text_blocks": texts}

    if ext in ["xlsx", "csv"]:
        df = pd.read_excel(file_path) if ext == "xlsx" else pd.read_csv(file_path)
        return {"type": "table", "tables": [df]}

    if ext in ["png", "jpg", "jpeg"]:
        return {"type": "image", "image": Image.open(file_path)}

    raise ValueError("Unsupported file format")


# -------- Normalization --------
def normalize(parsed, base_dir="assets"):
    os.makedirs(base_dir, exist_ok=True)

    texts, tables, images, formulas = [], [], [], []

    if parsed["type"] == "pdf":
        for t in parsed["text_blocks"]:
            texts.append({"text": t.text, "page": t.page_no})

        for i, table in enumerate(parsed["tables"]):
            df = table.to_dataframe().astype(str)
            flat = "\n".join([" | ".join(r) for r in df.values.tolist()])
            tables.append({"id": f"table_{i}", "text": flat, "page": table.page_no})

        os.makedirs(f"{base_dir}/figures", exist_ok=True)
        for i, fig in enumerate(parsed["figures"]):
            path = f"{base_dir}/figures/fig_{i}.png"
            fig.image.save(path)
            images.append({
                "id": f"fig_{i}",
                "path": path,
                "caption": fig.caption or "",
                "page": fig.page_no
            })

        for i, f in enumerate(parsed["formulas"]):
            formulas.append({
                "id": f"formula_{i}",
                "page": f.page_no,
                "latex": f"Equation on page {f.page_no} (OCR placeholder)"
            })

    elif parsed["type"] == "doc":
        for t in parsed["text_blocks"]:
            texts.append({"text": t, "page": None})

    elif parsed["type"] == "table":
        for i, df in enumerate(parsed["tables"]):
            flat = "\n".join([" | ".join(r) for r in df.astype(str).values.tolist()])
            tables.append({"id": f"table_doc_{i}", "text": flat, "page": None})

    elif parsed["type"] == "image":
        os.makedirs(f"{base_dir}/images", exist_ok=True)
        path = f"{base_dir}/images/img_0.png"
        parsed["image"].save(path)
        images.append({"id": "img_0", "path": path, "caption": "", "page": None})

    return texts, tables, images, formulas


# -------- Embeddings --------
def embed_text_items(items):
    vecs, meta = [], []
    for it in items:
        v = text_model.encode(it["text"], normalize_embeddings=True)
        vecs.append(v)
        meta.append(it)
    return np.array(vecs).astype("float32"), meta


def embed_image_items(items):
    vecs = []
    for img in items:
        im = preprocess(Image.open(img["path"])).unsqueeze(0).to(device)
        with torch.no_grad():
            v = clip_model.encode_image(im)
            v /= v.norm(dim=-1, keepdim=True)
        vecs.append(v.cpu().numpy()[0])
    return np.array(vecs).astype("float32")


# -------- FAISS --------
def build_index(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


# -------- Graph DB --------
def build_graph(text_meta, table_meta, images):
    G = nx.DiGraph()

    for t in text_meta:
        G.add_node(f"text_p{t['page']}", type="text")

    for img in images:
        G.add_node(img["id"], type="figure")
        if img["page"] is not None:
            G.add_edge(f"text_p{img['page']}", img["id"], relation="explains")

    for tbl in table_meta:
        G.add_node(tbl["id"], type="table")
        if tbl["page"] is not None:
            G.add_edge(tbl["id"], f"text_p{tbl['page']}", relation="supports")

    return G


# -------- Query Routing --------
def route(query):
    q = query.lower()
    if "equation" in q or "formula" in q:
        return "formula"
    if "table" in q:
        return "table"
    if "figure" in q or "chart" in q or "diagram" in q:
        return "image"
    return "text"


# -------- Search --------
def search_index(index, query, k=5):
    q = text_model.encode(query, normalize_embeddings=True)\
        .astype("float32").reshape(1, -1)
    _, I = index.search(q, k)
    return I[0]


# -------- Reasoning --------
def reason(context, question):
    prompt = f"""
Context:
{context}

Question:
{question}

Answer clearly with references.
"""
    return subprocess.check_output(
        ["ollama", "run", "gpt-oss:20b", prompt]
    ).decode()


# -------- Universal RAG --------
def universal_rag(
    query,
    text_index, table_index, formula_index, image_index,
    text_meta, table_meta, formula_meta, images
):
    r = route(query)

    if r == "text" and text_index is not None:
        ids = search_index(text_index, query)
        ctx = [text_meta[i]["text"] for i in ids]

    elif r == "table" and table_index is not None:
        ids = search_index(table_index, query)
        ctx = [table_meta[i]["text"] for i in ids]

    elif r == "formula" and formula_index is not None:
        ids = search_index(formula_index, query)
        ctx = [formula_meta[i]["latex"] for i in ids]

    elif r == "image" and image_index is not None:
        ids = search_index(image_index, query)
        ctx = [images[i]["caption"] for i in ids]

    else:
        ctx = []

    return reason("\n".join(ctx), query)


# -------- Example Usage --------
# parsed = ingest("your_complex_document.pdf")
# texts, tables, images, formulas = normalize(parsed)
#
# text_vecs, text_meta = embed_text_items(texts)
# table_vecs, table_meta = embed_text_items(tables)
# formula_vecs, formula_meta = embed_text_items(formulas)
# image_vecs = embed_image_items(images)
#
# text_index = build_index(text_vecs) if len(text_vecs) else None
# table_index = build_index(table_vecs) if len(table_vecs) else None
# formula_index = build_index(formula_vecs) if len(formula_vecs) else None
# image_index = build_index(image_vecs) if len(image_vecs) else None
#
# print(
#     universal_rag(
#         "Explain the performance improvement shown in the chart",
#         text_index, table_index, formula_index, image_index,
#         text_meta, table_meta, formula_meta, images
#     )
# )
