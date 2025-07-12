import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG PDF Assistant", layout="centered")
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

# --- Chat Interface ---
st.subheader("💬 Ask a Question")
question = st.text_input("Your question", "")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        response = requests.post(f"{API_URL}/ask", data={"question": question})
        if response.status_code == 200:
            data = response.json()
            st.session_state.chat_history.append({
                "question": question,
                "answer": data["answer"],
                "sources": data["sources"]
            })
        else:
            st.error("Something went wrong.")

# --- Show Chat History ---
if st.session_state.chat_history:
    st.subheader("📝 Chat History")
    for i, msg in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"**Q{i}:** {msg['question']}")
        st.markdown(f"**A{i}:** {msg['answer']}")
        st.markdown("---")

