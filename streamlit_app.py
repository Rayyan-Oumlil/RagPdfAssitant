import streamlit as st
import os
import tempfile
import time
from pathlib import Path
import sys
import threading
import requests
import json
import fitz  # PyMuPDF
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Page configuration
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode configuration
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Apply dark mode CSS
def apply_dark_mode():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        /* Main app background */
        .main .block-container {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #1e1e1e;
        }
        
        /* All text elements */
        .stMarkdown, .stText, .stWrite, .stSubheader, .stHeader {
            color: #ffffff !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #4CAF50 !important;
            color: white !important;
            border: 1px solid #4CAF50 !important;
        }
        
        .stButton > button:hover {
            background-color: #45a049 !important;
        }
        
        /* Text inputs */
        .stTextInput > div > div > input {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        
        /* Text areas */
        .stTextArea > div > div > textarea {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        
        /* Select boxes */
        .stSelectbox > div > div > select {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        
        /* File uploader */
        .stFileUploader > div > div {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        
        .streamlit-expanderContent {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }
        
        /* Metrics */
        .stMetric > div > div {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
        }
        
        /* Alerts and info boxes */
        .stAlert, .stInfo, .stWarning, .stError, .stSuccess {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        
        /* Toggle */
        .stToggle > label {
            color: #ffffff !important;
        }
        
        /* Links */
        a {
            color: #4CAF50 !important;
        }
        
        /* Code blocks */
        .stCodeBlock {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
        }
        
        /* Divider lines */
        hr {
            border-color: #555555 !important;
        }
        
        /* Main container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Force all text to be visible */
        * {
            color: inherit;
        }
        
        /* Specific override for any remaining invisible text */
        .stMarkdown p, .stMarkdown div, .stMarkdown span {
            color: #ffffff !important;
        }
        
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        /* Light mode - reset to defaults */
        .main .block-container {
            background-color: #ffffff;
            color: #31333f;
        }
        
        .css-1d391kg {
            background-color: #f0f2f6;
        }
        
        /* Reset all colors to default */
        .stMarkdown, .stText, .stWrite, .stSubheader, .stHeader {
            color: #31333f !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #31333f !important;
        }
        
        /* Reset form elements to default */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select,
        .stFileUploader > div > div {
            background-color: #ffffff !important;
            color: #31333f !important;
            border: 1px solid #d0d0d0 !important;
        }
        
        /* Reset buttons */
        .stButton > button {
            background-color: #ff4b4b !important;
            color: white !important;
        }
        
        /* Reset expanders */
        .streamlit-expanderHeader {
            background-color: #ffffff !important;
            color: #31333f !important;
        }
        
        .streamlit-expanderContent {
            background-color: #fafafa !important;
            color: #31333f !important;
        }
        
        /* Reset alerts */
        .stAlert, .stInfo, .stWarning, .stError, .stSuccess {
            background-color: #ffffff !important;
            color: #31333f !important;
        }
        
        /* Reset toggle */
        .stToggle > label {
            color: #31333f !important;
        }
        
        /* Reset links */
        a {
            color: #ff4b4b !important;
        }
        
        </style>
        """, unsafe_allow_html=True)

# Apply the theme
apply_dark_mode()

# Main title
st.title("🧠 AI Assistant to analyze your PDF documents")
st.markdown("**Powered by Streamlit Cloud - LLMs Cloud Support**")
st.markdown("---")

# Session variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False

# RAG Configuration
EMBEDDING_DIM = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Classes pour les LLMs Cloud
class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, api_key: str = None):
        # Try Streamlit secrets first, then environment variables
        if api_key:
            self.api_key = api_key
        else:
            # Try Streamlit secrets
            try:
                if self.__class__.__name__ == "GoogleProvider":
                    self.api_key = st.secrets.get("GOOGLE_API_KEY")
                elif self.__class__.__name__ == "HuggingFaceProvider":
                    self.api_key = st.secrets.get("HUGGINGFACE_API_KEY")
                elif self.__class__.__name__ == "OpenAIProvider":
                    self.api_key = st.secrets.get("OPENAI_API_KEY")
                elif self.__class__.__name__ == "AnthropicProvider":
                    self.api_key = st.secrets.get("ANTHROPIC_API_KEY")
                else:
                    self.api_key = None
            except:
                # Fallback to environment variables
                self.api_key = os.environ.get(f"{self.__class__.__name__.upper()}_API_KEY")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generates a response (to be implemented in child classes)"""
        raise NotImplementedError

class GoogleProvider(LLMProvider):
    """Google provider (Gemini)"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def generate_response(self, prompt: str, model: str = "gemini-1.5-flash", max_tokens: int = 500) -> str:
        """Generates a response with Google Gemini"""
        if not self.api_key:
            return "Error: Missing Google API key. Set GOOGLE_API_KEY in your environment variables."
        
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/models/{model}:generateContent?key={self.api_key}",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                return f"Google API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling Google: {str(e)}"

class HuggingFaceProvider(LLMProvider):
    """Hugging Face provider (open source models)"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api-inference.huggingface.co/models"
    
    def generate_response(self, prompt: str, model: str = "gpt2", max_length: int = 500) -> str:
        """Generates a response with Hugging Face"""
        if not self.api_key:
            return "Error: Missing Hugging Face API key. Set HUGGINGFACE_API_KEY in your environment variables."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_length": max_length,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").replace(prompt, "").strip()
                else:
                    return str(result)
            else:
                return f"Hugging Face API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling Hugging Face: {str(e)}"

class OpenAIProvider(LLMProvider):
    """OpenAI provider (GPT-3.5, GPT-4)"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.openai.com/v1"
    
    def generate_response(self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generates a response with OpenAI"""
        if not self.api_key:
            return "Error: Missing OpenAI API key. Set OPENAI_API_KEY in your environment variables."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an assistant that answers based only on the provided context. Answer concisely and accurately."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                return f"OpenAI API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"

class AnthropicProvider(LLMProvider):
    """Anthropic provider (Claude)"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.anthropic.com/v1"
    
    def generate_response(self, prompt: str, model: str = "claude-3-sonnet-20240229", max_tokens: int = 500) -> str:
        """Generates a response with Anthropic Claude"""
        if not self.api_key:
            return "Error: Missing Anthropic API key. Set ANTHROPIC_API_KEY in your environment variables."
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"].strip()
            else:
                return f"Anthropic API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling Anthropic: {str(e)}"

# Provider dictionary
LLM_PROVIDERS = {
    "google": GoogleProvider,
    "huggingface": HuggingFaceProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider
}

def get_llm_provider(provider_name: str):
    """Returns an instance of the requested LLM provider"""
    if provider_name.lower() in LLM_PROVIDERS:
        return LLM_PROVIDERS[provider_name.lower()]()
    return None

def check_provider_credentials(provider_name: str) -> bool:
    """Checks if credentials are available for a provider"""
    if provider_name.lower() not in LLM_PROVIDERS:
        return False
    
    # Try Streamlit secrets first
    try:
        if provider_name.lower() == "google" and st.secrets.get("GOOGLE_API_KEY"):
            return True
        elif provider_name.lower() == "huggingface" and st.secrets.get("HUGGINGFACE_API_KEY"):
            return True
        elif provider_name.lower() == "openai" and st.secrets.get("OPENAI_API_KEY"):
            return True
        elif provider_name.lower() == "anthropic" and st.secrets.get("ANTHROPIC_API_KEY"):
            return True
    except:
        pass
    
    # Fallback to environment variables
    env_var = f"{provider_name.upper()}_API_KEY"
    return bool(os.environ.get(env_var))

# Initialize RAG system
def initialize_rag():
    if not st.session_state.rag_initialized:
        # Create data folder if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Initialize global variables
        st.session_state.embed_model = load_embedding_model()
        st.session_state.index_path = "data/index.faiss"
        st.session_state.chunks_path = "data/doc_chunks.pkl"
        st.session_state.doc_chunks = []
        st.session_state.index = None
        
        # Load or create FAISS index
        if os.path.exists(st.session_state.index_path):
            st.session_state.index = faiss.read_index(st.session_state.index_path)
        else:
            st.session_state.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        
        # Load existing chunks
        if os.path.exists(st.session_state.chunks_path):
            with open(st.session_state.chunks_path, "rb") as f:
                st.session_state.doc_chunks = pickle.load(f)
        
        st.session_state.rag_initialized = True

# Utility functions
def extract_text_from_pdf(file) -> str:
    """Extracts text from a PDF file"""
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract text
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Clean up
        os.unlink(tmp_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def chunk_text(text: str) -> list:
    """Splits text into chunks"""
    if not text.strip():
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + CHUNK_SIZE
        
        if end < len(text):
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break
    
    return chunks

def add_document_to_index(file):
    """Adds a document to the index"""
    text = extract_text_from_pdf(file)
    if not text:
        return False, "Unable to extract text from PDF"
    
    chunks = chunk_text(text)
    if not chunks:
        return False, "No textual content found"
    
    # Add chunks
    st.session_state.doc_chunks.extend(chunks)
    
    # Create embeddings
    embeddings = st.session_state.embed_model.encode(chunks)
    st.session_state.index.add(np.array(embeddings))
    
    # Save
    faiss.write_index(st.session_state.index, st.session_state.index_path)
    with open(st.session_state.chunks_path, "wb") as f:
        pickle.dump(st.session_state.doc_chunks, f)
    
    return True, f"Document added with {len(chunks)} chunks"

def ask_question(question: str, model: str) -> dict:
    """Asks a question to the RAG system"""
    if not st.session_state.doc_chunks:
        return {
            "answer": "No documents are indexed. Please upload a PDF first.",
            "sources": []
        }
    
    # Semantic search
    question_vec = st.session_state.embed_model.encode([question])
    D, I = st.session_state.index.search(np.array(question_vec), 5)
    
    # Get sources
    sources = []
    for i in I[0]:
        if 0 <= i < len(st.session_state.doc_chunks):
            sources.append(st.session_state.doc_chunks[i])
    
    if not sources:
        return {
            "answer": "No relevant information found in the documents.",
            "sources": []
        }
    
    # Create context
    context = "\n\n".join(sources[:3])  # Limit to 3 sources
    
    # Use cloud LLM
    if model.startswith("cloud:"):
        provider_name = model.replace("cloud:", "")
        provider = get_llm_provider(provider_name)
        
        if provider:
            prompt = f"""You are a helpful assistant answering based only on the provided context.

Context:
{context}

Question: {question}

Give a short, direct answer. Do not add extra information not found in the context."""
            
            answer = provider.generate_response(prompt)
        else:
            answer = f"Provider {provider_name} not configured. Check your API keys."
    else:
        answer = f"Model {model} not recognized."
    
    return {
        "answer": answer,
        "sources": sources[:3]
    }

def reset_index():
    """Resets the index"""
    st.session_state.doc_chunks = []
    st.session_state.uploaded_files = []  # Clear uploaded files list
    st.session_state.index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
    # Delete files
    if os.path.exists(st.session_state.index_path):
        os.remove(st.session_state.index_path)
    if os.path.exists(st.session_state.chunks_path):
        os.remove(st.session_state.chunks_path)
    
    return True, "Index reset successfully"

def get_available_models():
    """Gets the list of available models"""
    models = []
    
    # Add configured cloud models
    for provider in ["google", "huggingface", "openai", "anthropic"]:
        if check_provider_credentials(provider):
            models.append(f"cloud:{provider}")
    
    return models

# Initialize RAG
initialize_rag()

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Dark mode toggle
    st.subheader("🎨 Theme")
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, help="Switch between light and dark themes")
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    st.markdown("---")
    
    # System status
    st.subheader("📊 System Status")
    doc_count = len(st.session_state.doc_chunks)
    st.write(f"Indexed documents: {doc_count}")
    
    if doc_count > 0:
        st.write(f"Total chunks: {len(st.session_state.doc_chunks)}")
        st.write(f"Index size: {st.session_state.index.ntotal}")
    
    # Available models
    st.subheader("🤖 Available Models")
    models = get_available_models()
    for model in models:
        provider = model.replace("cloud:", "")
        st.write(f"• {model} (LLM cloud)")
    
    if not models:
        st.warning("No cloud LLM configured. Add your API keys in the secrets.")
    
    # Actions
    st.subheader("🔄 Actions")
    if st.button("Reset Index", type="secondary"):
        success, message = reset_index()
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Select a PDF file to analyze"
    )
    
    if uploaded_file is not None:
        if st.button("📤 Upload Document", type="primary"):
            with st.spinner("Upload and indexing in progress..."):
                success, message = add_document_to_index(uploaded_file)
                if success:
                    st.success(message)
                    st.session_state.uploaded_files.append(uploaded_file.name)
                    st.rerun()
                else:
                    st.error(message)

with col2:
    st.header("📋 Uploaded Documents")
    if st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            st.write(f"• {file}")
    else:
        st.info("No documents uploaded")

# Chat area
st.header("💬 Chat with Assistant")

# Model selection
available_models = get_available_models()
if available_models:
    selected_model = st.selectbox(
        "Choose a model:",
        available_models,
        help="Select the model to use"
    )
else:
    st.error("No cloud LLM configured. Configure your API keys in Streamlit secrets.")
    st.stop()

# Question input area
question = st.text_area(
    "Ask your question:",
    placeholder="Ex: What is this document about? What are the key points?",
    height=100
)

# Button to ask question
if st.button("🤖 Ask Question", type="primary", disabled=not question.strip()):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching..."):
            response = ask_question(question, selected_model)
            
            # Add to history
            st.session_state.chat_history.append({
                "question": question,
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
                "model": selected_model
            })
            
            st.rerun()

# Display history
if st.session_state.chat_history:
    st.header("📝 Conversation History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            with st.expander(f"Question {len(st.session_state.chat_history) - i}: {chat['question'][:50]}..."):
                st.write(f"**Question:** {chat['question']}")
                st.write(f"**Model used:** {chat['model']}")
                st.write(f"**Answer:** {chat['answer']}")
                
                # Sources removed for cleaner interface
        
        with col2:
            # Delete button
            if st.button("🗑️", key=f"delete_{i}", help="Delete this question"):
                # Remove question from history
                index_to_delete = len(st.session_state.chat_history) - 1 - i
                st.session_state.chat_history.pop(index_to_delete)
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🧠 RAG PDF Assistant - Powered by Streamlit Cloud</p>
        <p>Analyze your PDF documents with AI and cloud LLMs</p>
    </div>
    """,
    unsafe_allow_html=True
) 