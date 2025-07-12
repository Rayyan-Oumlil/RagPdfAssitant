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

def ask_question(question: str, top_k: int = 5) -> str:
    question_vec = EMBED_MODEL.encode([question])
    D, I = index.search(np.array(question_vec), top_k)
    relevant_chunks = [DOC_CHUNKS[i] for i in I[0] if i < len(DOC_CHUNKS)]
    context = "\n\n".join(relevant_chunks)
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {question}"

    # Call local LLM (via Ollama CLI)
    result = subprocess.run(
        ['ollama', 'run', 'mistral'],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()
