# AI Research Assistant 🔬

**Evidence-Grounded AI Research Assistant** — A production-quality RAG (Retrieval-Augmented Generation) system that answers questions strictly from your uploaded documents.

## Features

- 📄 **Multi-format ingestion**: PDF, TXT, Markdown
- 🖼️ **Image understanding**: Gemini multimodal vision for charts, diagrams, scanned text
- 📊 **Table extraction**: PyMuPDF `find_tables()` → Markdown
- 🔍 **Hybrid retrieval**: Dense (BGE + ChromaDB) + Sparse (BM25) → RRF fusion
- 🎯 **Cross-encoder reranking**: `ms-marco-MiniLM-L-6-v2`
- 🤖 **Gemini LLM**: `gemini-3.8-flash` with strict grounding & JSON citations
- 🛡️ **Anti-hallucination**: Refuses if evidence is insufficient
- 🖥️ **Streamlit UI**: Chat, Sources panel, Evaluation log

## Architecture

```
Documents (PDF/TXT/MD)
       ↓
  DocumentParser (PyMuPDF)
       ↓
  TableExtractor + ImageProcessor (Gemini Vision)
       ↓
  Chunker (semantic + structural)
       ↓
  Embedder → BGE-small-en-v1.5 → ChromaDB
             + BM25 Index
                    ↓
  User Question
       ↓
  HybridRetriever (Dense + BM25 → RRF → Top-20)
       ↓
  Reranker (cross-encoder → Top-6, deduplicated)
       ↓
  ContextBuilder (metadata headers + injection guards)
       ↓
  LLMGenerator (gemini-3.8-flash → JSON: answer + citations)
       ↓
  Streamlit UI (Chat + Sources + Eval Log)
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key in .env
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Run the app
streamlit run app.py
```

## Project Structure

```
AI_Research_Assistant/
├── app.py                    # Streamlit UI
├── requirements.txt
├── .env                      # GEMINI_API_KEY
├── ingestion/
│   ├── document_parser.py    # PyMuPDF parsing
│   ├── image_processor.py    # Gemini vision OCR
│   ├── table_extractor.py    # Table → Markdown
│   ├── chunker.py            # Semantic chunking
│   └── embedder.py           # BGE + ChromaDB
├── retrieval/
│   ├── vector_store.py       # Dense retrieval
│   ├── bm25_retriever.py     # BM25 sparse
│   ├── hybrid_retriever.py   # RRF fusion
│   └── reranker.py           # Cross-encoder
├── generation/
│   ├── context_builder.py    # Prompt construction
│   └── llm_generator.py      # Gemini API
├── utils/
│   └── helpers.py
└── data/chroma_db/           # Vector store (auto-created)
```
