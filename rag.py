#!/usr/bin/env python3
"""RAG Engine — Retrieval-Augmented Generation for BlackRoad agents.

Uses Qdrant for vector storage and Ollama (nomic-embed-text) for embeddings.
"""

import hashlib
import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

import requests

QDRANT_URL = os.environ.get("QDRANT_URL", "http://192.168.4.49:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://192.168.4.96:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:1.5b")
COLLECTION = os.environ.get("RAG_COLLECTION", "blackroad_knowledge")
VECTOR_DIM = 768  # nomic-embed-text dimension


class RAGEngine:
    """Retrieval-Augmented Generation engine."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        ollama_url: str = OLLAMA_URL,
        collection: str = COLLECTION,
    ):
        self.qdrant_url = qdrant_url.rstrip("/")
        self.ollama_url = ollama_url.rstrip("/")
        self.collection = collection
        self._ensure_collection()

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            resp = requests.get(
                f"{self.qdrant_url}/collections/{self.collection}", timeout=5
            )
            if resp.status_code == 200:
                return
        except requests.ConnectionError:
            print(f"[rag] Warning: Qdrant not reachable at {self.qdrant_url}", file=sys.stderr)
            return

        requests.put(
            f"{self.qdrant_url}/collections/{self.collection}",
            json={
                "vectors": {
                    "size": VECTOR_DIM,
                    "distance": "Cosine",
                }
            },
            timeout=10,
        )

    def _embed(self, text: str) -> List[float]:
        """Get embedding vector from Ollama."""
        resp = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def _generate(self, prompt: str) -> str:
        """Generate text via Ollama."""
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def index(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Index a text chunk into the vector store. Returns the point ID."""
        point_id = str(uuid.uuid4())
        vector = self._embed(text)
        payload = {"text": text, **(metadata or {})}

        requests.put(
            f"{self.qdrant_url}/collections/{self.collection}/points",
            json={
                "points": [
                    {
                        "id": point_id,
                        "vector": vector,
                        "payload": payload,
                    }
                ]
            },
            timeout=10,
        )
        return point_id

    def index_file(self, filepath: str, chunk_size: int = 500) -> List[str]:
        """Index a text file, splitting into chunks."""
        with open(filepath, "r") as f:
            content = f.read()

        # Split into chunks by paragraphs or fixed size
        chunks = []
        paragraphs = content.split("\n\n")
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        ids = []
        for i, chunk in enumerate(chunks):
            pid = self.index(
                chunk,
                metadata={
                    "source": filepath,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            ids.append(pid)
            print(f"[rag] Indexed chunk {i+1}/{len(chunks)} from {filepath}", file=sys.stderr)
        return ids

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents. Returns list of {text, score, metadata}."""
        vector = self._embed(query)
        resp = requests.post(
            f"{self.qdrant_url}/collections/{self.collection}/points/search",
            json={
                "vector": vector,
                "limit": top_k,
                "with_payload": True,
            },
            timeout=10,
        )
        resp.raise_for_status()
        results = []
        for hit in resp.json().get("result", []):
            payload = hit.get("payload", {})
            results.append({
                "text": payload.pop("text", ""),
                "score": round(hit.get("score", 0), 4),
                "metadata": payload,
            })
        return results

    def ask(self, question: str, top_k: int = 5) -> Dict:
        """Search for context, then generate an answer via Ollama."""
        results = self.search(question, top_k=top_k)
        context = "\n\n---\n\n".join(r["text"] for r in results if r["text"])

        prompt = f"""Answer the question based on the following context. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""

        answer = self._generate(prompt)
        return {
            "question": question,
            "answer": answer.strip(),
            "sources": results,
            "num_sources": len(results),
        }

    def stats(self) -> Dict:
        """Get collection statistics."""
        try:
            resp = requests.get(
                f"{self.qdrant_url}/collections/{self.collection}", timeout=5
            )
            data = resp.json().get("result", {})
            return {
                "collection": self.collection,
                "points": data.get("points_count", 0),
                "vectors": data.get("vectors_count", 0),
                "status": data.get("status", "unknown"),
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    rag = RAGEngine()

    if len(sys.argv) < 2:
        print("Usage: python rag.py <command> [args]")
        print("Commands:")
        print("  index <text>           Index a text chunk")
        print("  index-file <path>      Index a text file (chunked)")
        print("  search <query>         Search for similar documents")
        print("  ask <question>         Search + generate answer")
        print("  stats                  Show collection stats")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "index" and len(sys.argv) >= 3:
        text = " ".join(sys.argv[2:])
        pid = rag.index(text)
        print(f"Indexed: {pid}")
    elif cmd == "index-file" and len(sys.argv) >= 3:
        ids = rag.index_file(sys.argv[2])
        print(f"Indexed {len(ids)} chunks")
    elif cmd == "search" and len(sys.argv) >= 3:
        query = " ".join(sys.argv[2:])
        for r in rag.search(query):
            print(f"  [{r['score']}] {r['text'][:100]}...")
    elif cmd == "ask" and len(sys.argv) >= 3:
        question = " ".join(sys.argv[2:])
        result = rag.ask(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"\n({result['num_sources']} sources used)")
    elif cmd == "stats":
        print(json.dumps(rag.stats(), indent=2))
    else:
        print("Unknown command")
