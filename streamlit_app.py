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

# Configuration de la page
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🧠 Assistant IA pour analyser vos documents PDF")
st.markdown("**Powered by Streamlit Cloud - LLMs Cloud Support**")
st.markdown("---")

# Variables de session
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False

# Configuration RAG
EMBEDDING_DIM = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialiser le modèle d'embedding
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Classes pour les LLMs Cloud
class LLMProvider:
    """Classe de base pour les fournisseurs de LLMs"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get(f"{self.__class__.__name__.upper()}_API_KEY")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Génère une réponse (à implémenter dans les classes enfants)"""
        raise NotImplementedError

class GoogleProvider(LLMProvider):
    """Fournisseur Google (Gemini)"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def generate_response(self, prompt: str, model: str = "gemini-1.5-flash", max_tokens: int = 500) -> str:
        """Génère une réponse avec Google Gemini"""
        if not self.api_key:
            return "Erreur: Clé API Google manquante. Définissez GOOGLE_API_KEY dans vos variables d'environnement."
        
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
                return f"Erreur Google API: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Erreur lors de l'appel à Google: {str(e)}"

class HuggingFaceProvider(LLMProvider):
    """Fournisseur Hugging Face (modèles open source)"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api-inference.huggingface.co/models"
    
    def generate_response(self, prompt: str, model: str = "gpt2", max_length: int = 500) -> str:
        """Génère une réponse avec Hugging Face"""
        if not self.api_key:
            return "Erreur: Clé API Hugging Face manquante. Définissez HUGGINGFACE_API_KEY dans vos variables d'environnement."
        
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
                return f"Erreur Hugging Face API: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Erreur lors de l'appel à Hugging Face: {str(e)}"

class OpenAIProvider(LLMProvider):
    """Fournisseur OpenAI (GPT-3.5, GPT-4)"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.openai.com/v1"
    
    def generate_response(self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Génère une réponse avec OpenAI"""
        if not self.api_key:
            return "Erreur: Clé API OpenAI manquante. Définissez OPENAI_API_KEY dans vos variables d'environnement."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Tu es un assistant qui répond uniquement basé sur le contexte fourni. Réponds de manière concise et précise."},
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
                return f"Erreur OpenAI API: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Erreur lors de l'appel à OpenAI: {str(e)}"

class AnthropicProvider(LLMProvider):
    """Fournisseur Anthropic (Claude)"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.anthropic.com/v1"
    
    def generate_response(self, prompt: str, model: str = "claude-3-sonnet-20240229", max_tokens: int = 500) -> str:
        """Génère une réponse avec Anthropic Claude"""
        if not self.api_key:
            return "Erreur: Clé API Anthropic manquante. Définissez ANTHROPIC_API_KEY dans vos variables d'environnement."
        
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
                return f"Erreur Anthropic API: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Erreur lors de l'appel à Anthropic: {str(e)}"

# Dictionnaire des fournisseurs
LLM_PROVIDERS = {
    "google": GoogleProvider,
    "huggingface": HuggingFaceProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider
}

def get_llm_provider(provider_name: str):
    """Retourne une instance du fournisseur LLM demandé"""
    if provider_name.lower() in LLM_PROVIDERS:
        return LLM_PROVIDERS[provider_name.lower()]()
    return None

def check_provider_credentials(provider_name: str) -> bool:
    """Vérifie si les credentials sont disponibles pour un fournisseur"""
    if provider_name.lower() not in LLM_PROVIDERS:
        return False
    
    env_var = f"{provider_name.upper()}_API_KEY"
    return bool(os.environ.get(env_var))

# Initialiser le système RAG
def initialize_rag():
    if not st.session_state.rag_initialized:
        # Créer le dossier data s'il n'existe pas
        os.makedirs("data", exist_ok=True)
        
        # Initialiser les variables globales
        st.session_state.embed_model = load_embedding_model()
        st.session_state.index_path = "data/index.faiss"
        st.session_state.chunks_path = "data/doc_chunks.pkl"
        st.session_state.doc_chunks = []
        st.session_state.index = None
        
        # Charger ou créer l'index FAISS
        if os.path.exists(st.session_state.index_path):
            st.session_state.index = faiss.read_index(st.session_state.index_path)
        else:
            st.session_state.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        
        # Charger les chunks existants
        if os.path.exists(st.session_state.chunks_path):
            with open(st.session_state.chunks_path, "rb") as f:
                st.session_state.doc_chunks = pickle.load(f)
        
        st.session_state.rag_initialized = True

# Fonctions utilitaires
def extract_text_from_pdf(file) -> str:
    """Extrait le texte d'un fichier PDF"""
    try:
        # Sauvegarder le fichier temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        # Extraire le texte
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Nettoyer
        os.unlink(tmp_path)
        return text
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du texte: {e}")
        return ""

def chunk_text(text: str) -> list:
    """Découpe le texte en chunks"""
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
    """Ajoute un document à l'index"""
    text = extract_text_from_pdf(file)
    if not text:
        return False, "Impossible d'extraire le texte du PDF"
    
    chunks = chunk_text(text)
    if not chunks:
        return False, "Aucun contenu textuel trouvé"
    
    # Ajouter les chunks
    st.session_state.doc_chunks.extend(chunks)
    
    # Créer les embeddings
    embeddings = st.session_state.embed_model.encode(chunks)
    st.session_state.index.add(np.array(embeddings))
    
    # Sauvegarder
    faiss.write_index(st.session_state.index, st.session_state.index_path)
    with open(st.session_state.chunks_path, "wb") as f:
        pickle.dump(st.session_state.doc_chunks, f)
    
    return True, f"Document ajouté avec {len(chunks)} chunks"

def ask_question(question: str, model: str = "local") -> dict:
    """Pose une question au système RAG"""
    if not st.session_state.doc_chunks:
        return {
            "answer": "Aucun document n'est indexé. Veuillez d'abord uploader un PDF.",
            "sources": []
        }
    
    # Recherche sémantique
    question_vec = st.session_state.embed_model.encode([question])
    D, I = st.session_state.index.search(np.array(question_vec), 5)
    
    # Récupérer les sources
    sources = []
    for i in I[0]:
        if 0 <= i < len(st.session_state.doc_chunks):
            sources.append(st.session_state.doc_chunks[i])
    
    if not sources:
        return {
            "answer": "Aucune information pertinente trouvée dans les documents.",
            "sources": []
        }
    
    # Créer le contexte
    context = "\n\n".join(sources[:3])  # Limiter à 3 sources
    
    # Générer la réponse
    if model == "local":
        # Réponse simple basée sur le contexte
        answer = f"Basé sur les documents, voici ce que j'ai trouvé :\n\n{context[:500]}..."
    elif model.startswith("cloud:"):
        # Utiliser un LLM cloud
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
            answer = f"Fournisseur {provider_name} non configuré. Vérifiez vos clés API."
    else:
        answer = f"Modèle {model} non reconnu."
    
    return {
        "answer": answer,
        "sources": sources[:3]
    }

def reset_index():
    """Réinitialise l'index"""
    st.session_state.doc_chunks = []
    st.session_state.index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
    # Supprimer les fichiers
    if os.path.exists(st.session_state.index_path):
        os.remove(st.session_state.index_path)
    if os.path.exists(st.session_state.chunks_path):
        os.remove(st.session_state.chunks_path)
    
    return True, "Index réinitialisé avec succès"

def get_available_models():
    """Récupère la liste des modèles disponibles"""
    models = ["local"]
    
    # Ajouter les modèles cloud configurés
    for provider in ["google", "huggingface", "openai", "anthropic"]:
        if check_provider_credentials(provider):
            models.append(f"cloud:{provider}")
    
    return models

# Initialiser le RAG
initialize_rag()

# Sidebar pour les contrôles
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Statut du système
    st.subheader("📊 Statut du système")
    doc_count = len(st.session_state.doc_chunks)
    st.write(f"Documents indexés: {doc_count}")
    
    if doc_count > 0:
        st.write(f"Chunks totaux: {len(st.session_state.doc_chunks)}")
        st.write(f"Taille de l'index: {st.session_state.index.ntotal}")
    
    # Modèles disponibles
    st.subheader("🤖 Modèles disponibles")
    models = get_available_models()
    for model in models:
        if model == "local":
            st.write(f"• {model} (recherche locale)")
        else:
            provider = model.replace("cloud:", "")
            st.write(f"• {model} (LLM cloud)")
    
    # Statut des clés API
    st.subheader("🔑 Clés API")
    api_keys = {
        "Google": "GOOGLE_API_KEY",
        "Hugging Face": "HUGGINGFACE_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY"
    }
    
    for name, key in api_keys.items():
        if os.environ.get(key):
            st.write(f"✅ {name}")
        else:
            st.write(f"❌ {name}")
    
    # Actions
    st.subheader("🔄 Actions")
    if st.button("Réinitialiser l'index", type="secondary"):
        success, message = reset_index()
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

# Zone principale
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📄 Upload de documents")
    
    uploaded_file = st.file_uploader(
        "Choisissez un fichier PDF",
        type=['pdf'],
        help="Sélectionnez un fichier PDF à analyser"
    )
    
    if uploaded_file is not None:
        if st.button("📤 Uploader le document", type="primary"):
            with st.spinner("Upload et indexation en cours..."):
                success, message = add_document_to_index(uploaded_file)
                if success:
                    st.success(message)
                    st.session_state.uploaded_files.append(uploaded_file.name)
                    st.rerun()
                else:
                    st.error(message)

with col2:
    st.header("📋 Documents uploadés")
    if st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            st.write(f"• {file}")
    else:
        st.info("Aucun document uploadé")

# Zone de chat
st.header("💬 Chat avec l'assistant")

# Sélection du modèle
available_models = get_available_models()
selected_model = st.selectbox(
    "Choisissez un modèle:",
    available_models,
    help="Sélectionnez le modèle à utiliser"
)

# Zone de saisie de question
question = st.text_area(
    "Posez votre question:",
    placeholder="Ex: De quoi parle ce document ? Quels sont les points clés ?",
    height=100
)

# Bouton pour poser la question
if st.button("🤖 Poser la question", type="primary", disabled=not question.strip()):
    if not question.strip():
        st.warning("Veuillez saisir une question.")
    else:
        with st.spinner("Recherche en cours..."):
            response = ask_question(question, selected_model)
            
            # Ajouter à l'historique
            st.session_state.chat_history.append({
                "question": question,
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
                "model": selected_model
            })
            
            st.rerun()

# Affichage de l'historique
if st.session_state.chat_history:
    st.header("📝 Historique des conversations")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Question {len(st.session_state.chat_history) - i}: {chat['question'][:50]}..."):
            st.write(f"**Question:** {chat['question']}")
            st.write(f"**Modèle utilisé:** {chat['model']}")
            st.write(f"**Réponse:** {chat['answer']}")
            
            if chat['sources']:
                st.write("**Sources:**")
                for j, source in enumerate(chat['sources'][:3]):
                    st.write(f"{j+1}. {source[:200]}...")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🧠 RAG PDF Assistant - Powered by Streamlit Cloud</p>
        <p>Analysez vos documents PDF avec l'IA et les LLMs cloud</p>
    </div>
    """,
    unsafe_allow_html=True
) 