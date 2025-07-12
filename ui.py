import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG PDF Assistant", layout="centered")
st.title("🧠 RAG PDF Assistant")

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

# --- Ask a Question ---
st.subheader("❓ Ask a Question")
question = st.text_input("Your question", "")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        response = requests.post(f"{API_URL}/ask", data={"question": question})
        if response.status_code == 200:
            st.success("✅ Answer:")
            st.write(response.json()["answer"])
        else:
            st.error("Something went wrong while asking the question.")
