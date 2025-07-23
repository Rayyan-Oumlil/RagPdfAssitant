from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from backend.utils import extract_text_from_pdf, chunk_text
import subprocess
import pickle
import requests

EMBEDDING_DIM = 384
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
INDEX_PATH = "data/index.faiss"
DOC_CHUNKS_PATH = "data/doc_chunks.pkl"
DOC_CHUNKS = []

# Create or load FAISS index
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)

# Charger les DOC_CHUNKS si le fichier existe
if os.path.exists(DOC_CHUNKS_PATH):
    with open(DOC_CHUNKS_PATH, "rb") as f:
        DOC_CHUNKS = pickle.load(f)

def add_document(file_path: str):
    global DOC_CHUNKS
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    DOC_CHUNKS.extend(chunks)
    embeddings = EMBED_MODEL.encode(chunks)
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_PATH)
    # Sauvegarder les DOC_CHUNKS
    with open(DOC_CHUNKS_PATH, "wb") as f:
        pickle.dump(DOC_CHUNKS, f)

def ask_question(question: str, model: str = "mistral", top_k: int = 5, threshold: float = 0.5) -> dict:
    question_vec = EMBED_MODEL.encode([question])
    D, I = index.search(np.array(question_vec), top_k)

    # Filter based on similarity score and valid indices
    valid_pairs = [(score, i) for score, i in zip(D[0], I[0]) if score < threshold and 0 <= i < len(DOC_CHUNKS)]
    sources = [DOC_CHUNKS[i] for score, i in valid_pairs]

    for score, i in valid_pairs:
        print(f"{score:.3f} → {DOC_CHUNKS[i][:80]}...")

    # Cas spécial : modèle openai et aucun document indexé => chatbot général
    if model == "openai" and len(DOC_CHUNKS) == 0:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"answer": "Clé API OpenAI manquante dans la variable d'environnement OPENAI_API_KEY.", "sources": []}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            "max_tokens": 512
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        if resp.status_code == 200:
            answer = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            answer = f"Erreur OpenAI API: {resp.text}"
        return {"answer": answer, "sources": []}

    if not sources:
        # Message explicite si base vide ou désynchronisation
        if len(DOC_CHUNKS) == 0 and (not hasattr(index, 'ntotal') or index.ntotal == 0):
            return {
                "answer": "Aucun document n’est indexé. Veuillez d’abord uploader un PDF.",
                "sources": []
            }
        else:
            return {
                "answer": "Désynchronisation détectée entre l'index et les documents. Veuillez réinitialiser l'index.",
                "sources": []
            }

    context = "\n\n".join(sources)
    prompt = f"""You are a helpful assistant answering based only on the provided context.

Context:
{context}

Question: {question}

Give a short, direct answer. Do not add extra information not found in the context.
"""

    if model == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"answer": "Clé API OpenAI manquante dans la variable d'environnement OPENAI_API_KEY.", "sources": sources}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        if resp.status_code == 200:
            answer = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            answer = f"Erreur OpenAI API: {resp.text}"
        return {"answer": answer, "sources": sources}

    print(f"🔁 Running with model: {model}")  # Optional log

    result = subprocess.run(
        ['ollama', 'run', model],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    answer = result.stdout.decode().strip()

    return {
        "answer": answer,
        "sources": sources
    }
