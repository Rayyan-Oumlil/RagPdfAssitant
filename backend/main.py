from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from backend.rag import add_document, ask_question

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

@app.post("/ask")
async def ask(question: str = Form(...)):
    answer = ask_question(question)
    return {"answer": answer}
