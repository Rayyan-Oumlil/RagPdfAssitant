from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from backend.utils import extract_text_from_pdf, chunk_text
import subprocess

EMBEDDING_DIM = 384
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
INDEX_PATH = "data/index.faiss"
DOC_CHUNKS = []

# Create or load FAISS index
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)

def add_document(file_path: str):
    global DOC_CHUNKS
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    DOC_CHUNKS.extend(chunks)
    embeddings = EMBED_MODEL.encode(chunks)
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_PATH)

def ask_question(question: str, top_k: int = 5, threshold: float = 0.5) -> dict:
    question_vec = EMBED_MODEL.encode([question])
    D, I = index.search(np.array(question_vec), top_k)

    # Filter based on similarity score
    sources = [DOC_CHUNKS[i] for score, i in zip(D[0], I[0]) if score < threshold]

    for score, i in zip(D[0], I[0]):
        print(f"{score:.3f} → {DOC_CHUNKS[i][:80]}...")


    if not sources:
        return {
            "answer": "Sorry, I couldn't find relevant information in the document.",
            "sources": []
        }

    context = "\n\n".join(sources)
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {question}"

    result = subprocess.run(
        ['ollama', 'run', 'mistral'],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )

    return {
        "answer": result.stdout.decode(),
        "sources": sources
    }

