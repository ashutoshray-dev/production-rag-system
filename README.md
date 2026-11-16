# Production RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions based on documents with **inline citations**. Built with LangGraph, Streamlit, and featuring hybrid retrieval (FAISS + BM25), cross-encoder re-ranking, and automated quality evaluation via Ragas.

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2.9-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-red.svg)

---

## Overview

This system implements a complete RAG pipeline with the following capabilities:

- **Document Q&A with Citations**: Answers questions using only provided context, with inline citations in format `[page no:X, source:filename]`
- **Hybrid Retrieval**: Combines dense vector search (FAISS) with sparse lexical search (BM25) via ensemble retrieval
- **Re-ranking**: Cross-encoder based re-ranking for improved relevance
- **Conversation Memory**: Persistent chat threads with SQLite checkpointing (LangGraph)
- **Multi-model Support**: Ollama (local), Google Gemini, OpenAI, and OpenRouter-compatible endpoints
- **Automated Evaluation**: Faithfulness scoring via Ragas with CI/CD quality gates
- **Streamlit UI**: Clean chat interface with thread management

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│  (Chat interface, thread management, message history)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph Pipeline                          │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │ pipeline_query node  │───▶│      chat_node               │  │
│  │ • Load/save FAISS    │    │ • Format docs with metadata  │  │
│  │ • Load/save BM25     │    │ • Apply system prompt        │  │
│  │ • Ensemble retrieve  │    │ • Generate cited response    │  │
│  │ • Cross-encoder rank │    │ • Auto-generate chat titles  │  │
│  └──────────────────────┘    └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌───────────┐    ┌────────────┐
       │  FAISS   │    │   BM25    │    │ SQLite DB  │
       │ (dense)  │    │ (sparse)  │    │(checkpoint)│
       └──────────┘    └───────────┘    └────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) (for local LLM) or API keys for cloud providers

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd production-rag-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull local model (if using Ollama)
ollama pull qwen2.5:3b
```

### Configuration

Copy the example environment file and configure your keys:

```bash
cp .env.examples .env
# Edit .env with your API keys
```

Required environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key | For Gemini models |
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI models |
| `OPENROUTER_API_KEY` | OpenRouter API key | For OpenRouter models |
| `HF_TOKEN` | HuggingFace token | For model downloads |
| `FREE_LLM_API_KEY` | Local/OpenRouter-compatible endpoint key | For model2 |
| `LANGCHAIN_API_KEY` | LangSmith API key | For tracing (optional) |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `true`/`false` |
| `LANGCHAIN_ENDPOINT` | LangSmith endpoint | Default: `https://api.smith.langchain.com` |
| `LANGCHAIN_PROJECT` | LangSmith project name | Default: `Production-RAG-system` |

### Running the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
production-rag-system/
├── app.py                      # Streamlit chat interface
├── requirements.txt            # Python dependencies
├── .env.examples               # Environment variable template
├── .env                        # Your environment variables (not committed)
├── rag-system.db               # SQLite checkpoint database (auto-created)
├── configs/
│   └── prompts.yaml            # System prompts (v1.0, v1.1)
├── backend/
│   ├── __init__.py
│   ├── models.py               # Model configurations & Pydantic schemas
│   ├── utils.py                # Document processing, retrieval, utilities
│   ├── graph.py                # LangGraph pipeline definition
│   └── checkpoint.py           # SQLite checkpointer setup
├── docs/
│   ├── ml_paper1.pdf           # Default knowledge base (ML survey paper)
│   └── nike_doc.pdf            # Additional document
├── index/                      # Vector & BM25 indices (auto-generated)
│   ├── faiss_*/                # FAISS indices per thread
│   └── bm25_*                  # BM25 pickles per thread
├── tests/
│   ├── __init__.py
│   ├── golden_set.json         # Evaluation benchmark (6 Q&A pairs)
│   ├── eval_ragas.py           # Ragas evaluation script
│   └── evaluation/
│       └── latest_eval_report.json  # Latest faithfulness scores
└── .github/
    └── workflows/
        └── ai_quality_gate.yml # CI/CD quality gate
```

---

## Key Features

### 1. Hybrid Retrieval (FAISS + BM25)

Combines dense semantic search with sparse keyword matching:

```python
# In backend/graph.py
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
bm25_retriever.k = 5
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)
```

### 2. Cross-Encoder Re-ranking

Improves retrieval precision using `cross-encoder/ms-marco-MiniLM-L-6-v2`:

```python
def rerank(retrieved_docs, query):
    sentence_pairs = [[query, doc.page_content] for doc in retrieved_docs]
    scores = encoder_model.predict(sentence_pairs)
    # Sort by relevance score
    return sorted(retrieved_docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
```

### 3. Citation-Enforced Generation

System prompt enforces strict citation format:

```yaml
# configs/prompts.yaml
system_prompt_v1.0: |
  For every statement you make, include an inline citation at the end of the sentence
  in this exact format: [page no:X, source: filename]
  If multiple sources support the statement, cite them all: [page no:1, source: doc1.pdf][page no:5, source: doc2.pdf]
```

### 4. Persistent Conversations

LangGraph SQLite checkpointer maintains chat history across sessions:

```python
# backend/checkpoint.py
conn = sqlite3.connect(database='rag-system.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
```

### 5. Multi-Model Flexibility

Configure models in `backend/models.py`:

```python
# Local (Ollama)
model = ChatOllama(model='qwen2.5:3b')

# Cloud (Gemini)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", ...)

# OpenRouter-compatible
model2 = ChatOpenAI(
    model="auto",
    openai_api_base="http://127.0.0.1:31415/v1",  # Local proxy or OpenRouter
    ...
)
```

---

## Evaluation & Quality Gates

### Running Evaluation Locally

```bash
# Requires OPENROUTER_API_KEY in .env
python -m tests.eval_ragas
```

### What Gets Evaluated

- **Faithfulness** (primary): Whether generated answers are grounded in retrieved context
- Uses `golden_set.json` with 6 benchmark questions from the ML paper
- Threshold: **0.75** (configurable in `eval_ragas.py`)

### CI/CD Pipeline

GitHub Actions runs evaluation on every push/PR to `main`:

```yaml
# .github/workflows/ai_quality_gate.yml
- name: Execute Offline Ragas Benchmark
  env:
    GOOGLE_API_KEY: ${{secrets.GOOGLE_API_KEY}}
  run: python -m tests.eval_ragas
```

Fails the build if faithfulness drops below threshold.

### Latest Results

```json
// tests/evaluation/latest_eval_report.json
{
  "faithfulness": 0.9
}
```

---

## Adding Documents

1. Place PDF files in `docs/`
2. Update the default path in `backend/graph.py` (line 38) or pass `path` parameter:
   ```python
   # Default
   def pipeline_query(state: ChatState, config: RunnableConfig, path='docs/ml_paper1.pdf'):
   ```
3. Indices are auto-generated per conversation thread on first query

---

## API Reference

### Core Components

| Module | Purpose |
|--------|---------|
| `backend.models` | LLM/embedding configs, Pydantic schemas (`ChatTitle`, `FinalResponse`) |
| `backend.utils` | Document loading, splitting, retrieval, thread management, prompt loading |
| `backend.graph` | LangGraph pipeline (`pipeline_query` → `chat_node`) |
| `backend.checkpoint` | SQLite checkpointer for conversation persistence |

### State Schema

```python
# backend/graph.py
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    chat_title: str
    docs: list[Document]
    vector_store_path: str
    bm25_path: str
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FAISS load_local` error | Ensure `allow_dangerous_deserialization=True` (already set) |
| BM25 pickle errors | Delete `index/bm25_*` files to regenerate |
| Model connection refused | Check Ollama is running (`ollama serve`) or API keys are valid |
| Low faithfulness scores | Adjust retrieval `k`, re-ranking, or prompt in `configs/prompts.yaml` |
| Thread not loading | Check `rag-system.db` exists and isn't corrupted |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run evaluation: `python -m tests.eval_ragas`
4. Ensure CI passes
5. Submit a PR

---

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Ragas](https://github.com/explodinggradients/ragas) for evaluation
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [FastEmbed](https://github.com/qdrant/fastembed) for lightweight embeddings