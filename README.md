# 🧠 RAG PDF Assistant

Un assistant IA intelligent pour analyser vos documents PDF en utilisant la technologie RAG (Retrieval-Augmented Generation) et les LLMs cloud sur Streamlit Cloud.

## ✨ Fonctionnalités

- 📄 **Upload PDF** : Interface drag & drop pour uploader vos documents
- 🤖 **LLMs Cloud** : Support complet pour Google Gemini, Hugging Face, OpenAI, Anthropic
- 🔍 **RAG System** : Recherche sémantique dans vos documents
- 💬 **Chat intelligent** : Posez des questions sur vos documents
- 📊 **Interface moderne** : Design responsive avec Streamlit
- 📝 **Historique** : Conversations sauvegardées
- 🚀 **Déploiement cloud** : Prêt pour Streamlit Cloud

## 🚀 Déploiement

### Streamlit Cloud (Recommandé)

1. **Forkez ce repository** sur GitHub
2. **Allez sur** [share.streamlit.io](https://share.streamlit.io)
3. **Connectez-vous** avec votre compte GitHub
4. **Créez une nouvelle app** :
   - **Repository** : `votre-username/rag-assistant`
   - **Branch** : `main`
   - **Main file path** : `streamlit_app.py`
5. **Configurez les variables d'environnement** (voir section Configuration)
6. **Cliquez sur "Deploy"**

### Local

```bash
# Installation
pip install -r requirements.txt

# Démarrage
streamlit run streamlit_app.py
```

## 🎯 Utilisation

1. **Upload** un fichier PDF
2. **Sélectionnez** un modèle (local ou cloud)
3. **Posez** vos questions
4. **Consultez** les réponses et sources

## 📁 Structure

```
rag-assistant/
├── streamlit_app.py          # Application principale
├── requirements.txt          # Dépendances
├── .streamlit/
│   └── config.toml          # Configuration Streamlit
├── README.md                # Ce fichier
└── .gitignore              # Fichiers à ignorer
```

## 🛠️ Technologies

- **Streamlit** : Interface web
- **PyMuPDF** : Traitement PDF
- **SentenceTransformers** : Embeddings
- **FAISS** : Index vectoriel
- **NumPy** : Calculs numériques
- **LLMs Cloud** : Google, Hugging Face, OpenAI, Anthropic

## 🔧 Configuration

### Variables d'environnement

Pour utiliser les LLMs cloud, configurez ces variables dans Streamlit Cloud :

```bash
# Google Gemini
GOOGLE_API_KEY=votre_clé_google

# Hugging Face
HUGGINGFACE_API_KEY=votre_clé_huggingface

# OpenAI
OPENAI_API_KEY=votre_clé_openai

# Anthropic Claude
ANTHROPIC_API_KEY=votre_clé_anthropic
```

### Comment obtenir les clés API

- **Google Gemini** : [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Hugging Face** : [Hugging Face Settings](https://huggingface.co/settings/tokens)
- **OpenAI** : [OpenAI Platform](https://platform.openai.com/api-keys)
- **Anthropic** : [Anthropic Console](https://console.anthropic.com/)

## 📊 Fonctionnalités RAG

- **Extraction de texte** : PyMuPDF pour les PDFs
- **Chunking** : Découpage intelligent en morceaux
- **Embeddings** : SentenceTransformers (all-MiniLM-L6-v2)
- **Index vectoriel** : FAISS pour la recherche rapide
- **Recherche sémantique** : Similarité cosinus
- **LLMs Cloud** : Génération de réponses intelligentes

## 🤖 LLMs Supportés

### Mode Local
- **Recherche locale** : Basé sur la similarité sémantique

### Mode Cloud
- **Google Gemini** : `gemini-1.5-flash`
- **Hugging Face** : Modèles open source (gpt2, etc.)
- **OpenAI** : GPT-3.5-turbo, GPT-4
- **Anthropic** : Claude-3-sonnet

## 🎨 Interface

- **Design moderne** : Interface intuitive et responsive
- **Upload drag & drop** : Glissez-déposez vos PDFs
- **Chat interactif** : Posez des questions naturellement
- **Historique** : Consultez vos conversations précédentes
- **Statut en temps réel** : Suivez l'état de votre index
- **Statut des clés API** : Vérifiez la configuration

## 🚀 Performance

- **Indexation rapide** : FAISS pour la recherche ultra-rapide
- **Cache intelligent** : Modèles chargés une seule fois
- **Optimisation mémoire** : Gestion efficace des ressources
- **Upload optimisé** : Traitement asynchrone des fichiers
- **LLMs cloud** : Réponses de haute qualité

## 🔒 Sécurité

- **Variables d'environnement** : Clés API sécurisées
- **Validation des fichiers** : Vérification des types PDF
- **Limites de taille** : Protection contre les fichiers trop gros
- **Isolation** : Environnements séparés

## 🛠️ Dépannage

### Problèmes courants

1. **"Clé API manquante"**
   - Vérifiez les variables d'environnement dans Streamlit Cloud
   - Assurez-vous que les clés sont correctes

2. **"Modèle non disponible"**
   - Vérifiez que la clé API correspondante est configurée
   - Redéployez après modification des variables

3. **"Erreur d'upload"**
   - Vérifiez que le fichier est un PDF valide
   - Assurez-vous que la taille ne dépasse pas 200MB

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Ouvrir une issue pour signaler un bug
- Proposer une amélioration
- Soumettre une pull request

## 📄 Licence

MIT License

---

**🎉 Votre assistant RAG avec LLMs cloud est maintenant prêt pour Streamlit Cloud !** 