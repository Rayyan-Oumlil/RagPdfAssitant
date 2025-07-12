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

| Category | Tools Used |
|----------|------------|
| Backend  | **FastAPI**, Python |
| AI / NLP | **Mistral (via Ollama)**, SentenceTransformers |
| Embeddings | `all-MiniLM-L6-v2` |
| Vector Search | **FAISS** |
| File Parsing | **PyMuPDF** |
| Deployment | Localhost, Docker (optional) |
| UI (Optional) | Streamlit or React |

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
