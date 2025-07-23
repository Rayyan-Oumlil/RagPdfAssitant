from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from backend.rag import add_document, ask_question
import subprocess

app = FastAPI()

# Enable CORS for Streamlit or frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "data"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    add_document(file_path)
    return {"message": "File uploaded and indexed successfully"}

from fastapi import Form

@app.post("/ask")
async def ask(question: str = Form(...), model: str = Form("mistral")):
    try:
        response = ask_question(question, model=model)
        return response
    except Exception as e:
        return {"answer": "Une erreur est survenue : " + str(e), "sources": []}

# Nouvelle route pour lister les modèles Ollama installés
@app.get("/models")
async def list_models():
    result = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode()
    # La sortie ressemble à : "NAME            SIZE   ...\nmistral         4.1G   ...\ndeepseek-coder  8.0G   ..."
    lines = output.strip().split("\n")[1:]  # On saute l'en-tête
    models = [line.split()[0] for line in lines if line.strip()]
    return {"models": models}

@app.post("/reset")
async def reset_index():
    index_path = os.path.join(UPLOAD_FOLDER, "index.faiss")
    chunks_path = os.path.join(UPLOAD_FOLDER, "doc_chunks.pkl")
    removed = []
    for path in [index_path, chunks_path]:
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    # Réinitialiser les variables globales en mémoire
    from backend import rag
    rag.DOC_CHUNKS.clear()
    rag.index = rag.faiss.IndexFlatL2(rag.EMBEDDING_DIM)
    return {"message": f"Fichiers supprimés et mémoire réinitialisée : {removed}"}


