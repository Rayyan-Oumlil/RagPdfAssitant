# 🧠 RAG PDF Assistant

An intelligent AI assistant to analyze your PDF documents using RAG (Retrieval-Augmented Generation) technology and cloud LLMs on Streamlit Cloud.

## ✨ Features

- 📄 **PDF Upload** : Drag & drop interface to upload your documents
- 🤖 **Cloud LLMs** : Full support for Google Gemini, Hugging Face, OpenAI, Anthropic
- 🔍 **RAG System** : Semantic search in your documents
- 💬 **Smart Chat** : Ask questions about your documents
- 📊 **Modern Interface** : Responsive design with Streamlit
- 📝 **History** : Saved conversations
- 🚀 **Cloud Deployment** : Ready for Streamlit Cloud

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. **Fork this repository** on GitHub
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Sign in** with your GitHub account
4. **Create a new app**:
   - **Repository** : `your-username/rag-assistant`
   - **Branch** : `main`
   - **Main file path** : `streamlit_app.py`
5. **Configure environment variables** (see Configuration section)
6. **Click "Deploy"**

### Local

```bash
# Installation
pip install -r requirements.txt

# Start
streamlit run streamlit_app.py
```

## 🎯 Usage

1. **Upload** a PDF file
2. **Select** a model (local or cloud)
3. **Ask** your questions
4. **View** answers and sources

## 📁 Structure

```
rag-assistant/
├── streamlit_app.py          # Main application
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── README.md                # This file
└── .gitignore              # Files to ignore
```

## 🛠️ Technologies

- **Streamlit** : Web interface
- **PyMuPDF** : PDF processing
- **SentenceTransformers** : Embeddings
- **FAISS** : Vector index
- **NumPy** : Numerical computations
- **Cloud LLMs** : Google, Hugging Face, OpenAI, Anthropic

## 🔧 Configuration

### Environment Variables

To use cloud LLMs, configure these variables in Streamlit Cloud:

```bash
# Google Gemini
GOOGLE_API_KEY=your_google_key

# Hugging Face
HUGGINGFACE_API_KEY=your_huggingface_key

# OpenAI
OPENAI_API_KEY=your_openai_key

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_key
```

### How to get API keys

- **Google Gemini** : [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Hugging Face** : [Hugging Face Settings](https://huggingface.co/settings/tokens)
- **OpenAI** : [OpenAI Platform](https://platform.openai.com/api-keys)
- **Anthropic** : [Anthropic Console](https://console.anthropic.com/)

## 📊 RAG Features

- **Text extraction** : PyMuPDF for PDFs
- **Chunking** : Intelligent text splitting
- **Embeddings** : SentenceTransformers (all-MiniLM-L6-v2)
- **Vector index** : FAISS for fast search
- **Semantic search** : Cosine similarity
- **Cloud LLMs** : Intelligent response generation

## 🤖 Supported LLMs

### Local Mode
- **Local search** : Based on semantic similarity

### Cloud Mode
- **Google Gemini** : `gemini-1.5-flash`
- **Hugging Face** : Open source models (gpt2, etc.)
- **OpenAI** : GPT-3.5-turbo, GPT-4
- **Anthropic** : Claude-3-sonnet

## 🎨 Interface

- **Modern design** : Intuitive and responsive interface
- **Drag & drop upload** : Drag and drop your PDFs
- **Interactive chat** : Ask questions naturally
- **History** : View your previous conversations
- **Real-time status** : Track your index status
- **API key status** : Check configuration

## 🚀 Performance

- **Fast indexing** : FAISS for ultra-fast search
- **Smart caching** : Models loaded only once
- **Memory optimization** : Efficient resource management
- **Optimized upload** : Asynchronous file processing
- **Cloud LLMs** : High-quality responses

## 🔒 Security

- **Environment variables** : Secure API keys
- **File validation** : PDF type verification
- **Size limits** : Protection against oversized files
- **Isolation** : Separate environments

## 🛠️ Troubleshooting

### Common issues

1. **"Missing API key"**
   - Check environment variables in Streamlit Cloud
   - Make sure keys are correct

2. **"Model not available"**
   - Verify that the corresponding API key is configured
   - Redeploy after modifying variables

3. **"Upload error"**
   - Check that the file is a valid PDF
   - Make sure size doesn't exceed 200MB

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open an issue to report a bug
- Propose an improvement
- Submit a pull request

## 📄 License

MIT License

---

**🎉 Your RAG assistant with cloud LLMs is now ready for Streamlit Cloud!** 