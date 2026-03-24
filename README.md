# rag-engine

Retrieval-Augmented Generation pipeline for BlackRoad agents. Qdrant vector search + Ollama embeddings and generation.

## Setup

```bash
pip install -r requirements.txt
```

Requires:
- Qdrant running at `QDRANT_URL` (default: `http://192.168.4.49:6333` on Alice)
- Ollama running at `OLLAMA_URL` (default: `http://192.168.4.96:11434` on Cecilia)
- Models: `nomic-embed-text` (embeddings), `qwen2.5:1.5b` (generation)

## Usage

```python
from rag import RAGEngine

rag = RAGEngine()

# Index documents
rag.index("BlackRoad OS runs on 5 Raspberry Pis", {"source": "docs"})
rag.index_file("knowledge.txt")  # auto-chunks

# Search
results = rag.search("how many pis", top_k=5)

# Ask (search + generate)
answer = rag.ask("What hardware does BlackRoad use?")
print(answer["answer"])
```

## CLI

```bash
python rag.py index "text to index"
python rag.py index-file knowledge.txt
python rag.py search "query"
python rag.py ask "question"
python rag.py stats
```

## Part of BlackRoad-Agents

Remember the Road. Pave Tomorrow.

BlackRoad OS, Inc. — Incorporated 2025.
