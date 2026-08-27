import os
import glob
from contextlib import asynccontextmanager

import pymupdf  # PyMuPDF pour les PDF
import docx
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Modèle d'embedding léger (384 dimensions)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Structure de stockage en mémoire pour le lab
vector_store = []

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".pdf":
        doc = pymupdf.open(file_path)
        for page in doc:
            text += page.get_text() + "\n"
    elif ext == ".docx":
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text])
    elif ext == ".md":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks

def index_dataset():
    dataset_path = "./datasets"
    files = glob.glob(f"{dataset_path}/*.*")

    for fpath in files:
        fname = os.path.basename(fpath)
        raw_text = extract_text(fpath)
        chunks = chunk_text(raw_text)

        for idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            vector_store.append({
                "id": f"{fname}_chunk_{idx}",
                "filename": fname,
                "text": chunk,
                "vector": embedding
            })
    print(f"Indexation terminée : {len(vector_store)} chunks créés.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    index_dataset()
    yield

app = FastAPI(title="Lab Embedding Engine", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/search")
def search(req: QueryRequest):
    query_vec = model.encode(req.query)

    results = []
    for item in vector_store:
        doc_vec = np.array(item["vector"])
        # Cosine similarity
        score = float(np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)))
        results.append({
            "filename": item["filename"],
            "text": item["text"],
            "score": score
        })

    # Tri par score décroissant
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"query": req.query, "results": results[:req.top_k]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5050, reload=True)