<!-- BlackRoad SEO Enhanced -->

# rag engine

> Part of **[BlackRoad OS](https://blackroad.io)** — Sovereign Computing for Everyone

[![BlackRoad OS](https://img.shields.io/badge/BlackRoad-OS-ff1d6c?style=for-the-badge)](https://blackroad.io)
[![BlackRoad-Agents](https://img.shields.io/badge/Org-BlackRoad-Agents-2979ff?style=for-the-badge)](https://github.com/BlackRoad-Agents)

**rag engine** is part of the **BlackRoad OS** ecosystem — a sovereign, distributed operating system built on edge computing, local AI, and mesh networking by **BlackRoad OS, Inc.**

### BlackRoad Ecosystem
| Org | Focus |
|---|---|
| [BlackRoad OS](https://github.com/BlackRoad-OS) | Core platform |
| [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc) | Corporate |
| [BlackRoad AI](https://github.com/BlackRoad-AI) | AI/ML |
| [BlackRoad Hardware](https://github.com/BlackRoad-Hardware) | Edge hardware |
| [BlackRoad Security](https://github.com/BlackRoad-Security) | Cybersecurity |
| [BlackRoad Quantum](https://github.com/BlackRoad-Quantum) | Quantum computing |
| [BlackRoad Agents](https://github.com/BlackRoad-Agents) | AI agents |
| [BlackRoad Network](https://github.com/BlackRoad-Network) | Mesh networking |

**Website**: [blackroad.io](https://blackroad.io) | **Chat**: [chat.blackroad.io](https://chat.blackroad.io) | **Search**: [search.blackroad.io](https://search.blackroad.io)

---


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
