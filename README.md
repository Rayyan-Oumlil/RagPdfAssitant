# 🧠 RAG PDF Assistant

A private AI assistant that lets you upload PDF files and ask questions about them using Retrieval-Augmented Generation (RAG) and local LLMs via Ollama.

## 🔧 Tech Stack
- FastAPI (backend)
- PyMuPDF (PDF parsing)
- SentenceTransformers (embeddings)
- FAISS (vector search)
- Ollama + Mistral (local LLM)
- (Optional) Streamlit or React (UI)

## 🚀 How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn backend.main:app --reload

# In another terminal, run the LLM
ollama run mistral
