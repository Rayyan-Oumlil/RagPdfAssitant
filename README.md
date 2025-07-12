# 🧠 RAG PDF Assistant

A local AI assistant that lets you upload **PDF files** and ask **questions** about them using **RAG (Retrieval-Augmented Generation)**. It runs **fully offline** using a local LLM like Mistral via [Ollama](https://ollama.com), and uses `FAISS` + `sentence-transformers` for semantic search.

> Built with FastAPI, PyMuPDF, FAISS, and Mistral LLM.

---

## 📦 Features

- 🔍 Extracts text from PDF files
- 🧠 Splits and embeds document chunks
- 🔎 Retrieves relevant content using vector search (FAISS)
- 🤖 Answers questions using a local LLM (Mistral)
- 💾 Works fully offline (no OpenAI API needed)
- 🔧 Easy to extend with Streamlit, React, or Docker

---

## 🚀 Tech Stack

| Category   | Tools Used                           |
|------------|---------------------------------------|
| Backend    | FastAPI, Python                      |
| AI / NLP   | Mistral (via Ollama), SentenceTransformers |
| Embeddings | all-MiniLM-L6-v2                     |
| Vector DB  | FAISS                                |
| File I/O   | PyMuPDF                              |
| Optional   | Streamlit or React (for UI)          |

---

## 📁 Project Structure

rag-assistant/
├── backend/ # FastAPI backend
│ ├── main.py # API endpoints
│ ├── rag.py # RAG logic (embedding + retrieval)
│ └── utils.py # PDF parsing + chunking
├── data/ # Uploaded files & FAISS index
├── models/ # Optional LLM/config files
├── requirements.txt # Python dependencies
├── .gitignore # Ignore cache, env, FAISS index
└── README.md # You're here!

yaml
Copier
Modifier

---

## ⚙️ Requirements

- Python 3.10+
- [Ollama](https://ollama.com) (for local LLMs)
- Git, pip

---

## 🧪 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/Rayyan-Oumlil/RagPdfAssitant.git
cd RagPdfAssitant
2. Create a Virtual Environment (optional)
bash
Copier
Modifier
python -m venv venv
venv\Scripts\activate  # Windows
3. Install Dependencies
bash
Copier
Modifier
pip install -r requirements.txt
4. Start Ollama + Download the Mistral Model
bash
Copier
Modifier
ollama run mistral
The first time, it will download ~4.1GB. Keep this terminal open.

5. Start the FastAPI Server
Open another terminal and run:

bash
Copier
Modifier
uvicorn backend.main:app --reload
Visit the docs at:
📎 http://127.0.0.1:8000/docs

📡 API Endpoints
➕ POST /upload
Upload and index a PDF file

Form field: file: UploadFile

❓ POST /ask
Ask a question about the uploaded content

Form field: question: str

🧠 How It Works (RAG Flow)
Upload a PDF

Text is extracted and split into small chunks

Chunks are embedded with sentence-transformers

Embeddings stored in FAISS

When a question is asked:

It's embedded

Top chunks are retrieved from FAISS

Chunks + question are passed to the LLM (Mistral via Ollama)

A natural language answer is generated

🧱 Python Dependencies
nginx
Copier
Modifier
fastapi
uvicorn
pymupdf
sentence-transformers
faiss-cpu
python-multipart
🛡️ Limitations
Not production-ready (no auth, rate-limiting, or sanitization)

Currently only supports .pdf files

Mistral output quality may vary; no fine-tuning applied

✅ TODO / Improvements
 Add Streamlit or React UI

 Add citations and source highlighting

 Support multiple documents / users

 Dockerize the project

 Add .txt file support

👨‍💻 Author
Built by Rayyan Oumlil

📝 License
MIT License. Free to use, modify, and share.

yaml
Copier
Modifier

---

Once you add this file to your repo and commit:

```bash
git add README.md
git commit -m "Add full README"
git push origin main
