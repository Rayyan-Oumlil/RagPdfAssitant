import streamlit as st
import requests
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG PDF Assistant", layout="centered")

# --- Ajout d'un background dégradé animé ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(270deg, #ffecd2, #fcb69f, #a1c4fd, #c2e9fb, #fcb69f, #ffecd2);
        background-size: 1200% 1200%;
        animation: gradientBG 20s ease infinite;
        min-height: 100vh;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 RAG PDF Assistant")

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- File Upload ---
st.subheader("📤 Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Uploading..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_URL}/upload", files={"file": (uploaded_file.name, uploaded_file, "application/pdf")})
        if response.status_code == 200:
            st.success("File uploaded and indexed! ✅")
        else:
            st.error("Failed to upload file 😞")

st.subheader("🧠 Choose LLM Model")
# Récupération dynamique des modèles installés
try:
    resp = requests.get(f"{API_URL}/models")
    if resp.status_code == 200:
        models = resp.json().get("models", [])
        if "openai" not in models:
            models.append("openai")
        if not models:
            st.warning("Aucun modèle Ollama détecté. Veuillez en installer un.")
    else:
        models = ["mistral", "openai"]
        st.warning("Impossible de récupérer la liste des modèles. Utilisation de 'mistral' et 'openai' par défaut.")
except Exception as e:
    models = ["mistral", "openai"]
    st.warning(f"Erreur lors de la récupération des modèles: {e}")

model_name = st.selectbox("Select a local LLM (must be installed in Ollama) or OpenAI API:", models, index=0)

# --- Chat Interface ---
st.subheader("💬 Ask a Question")
question = st.text_input("Your question", "")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        response = requests.post(f"{API_URL}/ask", data={"question": question, "model": model_name})
        if response.status_code == 200:
            data = response.json()
            # Affichage d'une erreur explicite si présente
            if data.get("answer", "").startswith("Une erreur est survenue"):
                st.error(data["answer"])
            else:
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": data["answer"],
                    "sources": data["sources"]
                })
        else:
            st.error("Something went wrong.")

# --- Bouton de réinitialisation de la base ---
if st.button("🧹 Réinitialiser la base (index + documents)"):
    resp = requests.post(f"{API_URL}/reset")
    if resp.status_code == 200:
        st.success("Base réinitialisée. Vous pouvez réuploader un PDF.")
        st.session_state.chat_history = []
    else:
        st.error("Erreur lors de la réinitialisation.")

# --- Show Chat History ---
if st.session_state.chat_history:
    st.subheader("📝 Chat History")
    for i, msg in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"**Q{i}:** {msg['question']}")
        st.markdown(f"**A{i}:** {msg['answer']}")
        st.markdown("---")

