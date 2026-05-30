# rag/rag_pipeline.py
# ============================================================
# Full RAG pipeline: ingest → embed → store → retrieve
# ============================================================

import hashlib
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)
from utils.logger import logger
from utils.text_utils import clean_text, chunk_text


class RAGPipeline:
    """
    End-to-end RAG pipeline using ChromaDB + SentenceTransformers.

    Usage:
        rag = RAGPipeline()
        rag.ingest_documents(documents)
        context = rag.retrieve("What is the pricing model?")
    """

    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME):
        self.collection_name = collection_name
        self._init_embedder()
        self._init_chroma()

    # ── Initialization ────────────────────────────────────────

    def _init_embedder(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def _init_chroma(self):
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{self.collection_name}' ready. "
                    f"Docs: {self.collection.count()}")

    # ── Ingestion ─────────────────────────────────────────────

    def _doc_id(self, text: str, source: str) -> str:
        """Deterministic ID to avoid duplicates."""
        return hashlib.sha256(f"{source}::{text[:200]}".encode()).hexdigest()[:32]

    def ingest_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        Ingest a list of documents.
        Each document: {"text": str, "source": str, "title": str (optional)}

        Returns number of chunks stored.
        """
        chunks_added = 0

        for doc in documents:
            raw_text = doc.get("text", "")
            source = doc.get("source", "unknown")
            title = doc.get("title", "")

            if not raw_text or len(raw_text) < 50:
                continue

            cleaned = clean_text(raw_text)
            chunks = chunk_text(cleaned, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

            ids, embeddings_list, metadatas, texts = [], [], [], []

            for i, chunk in enumerate(chunks):
                doc_id = self._doc_id(chunk, f"{source}_{i}")

                # Skip if already exists
                existing = self.collection.get(ids=[doc_id])
                if existing["ids"]:
                    continue

                embedding = self.embedder.encode(chunk, normalize_embeddings=True).tolist()

                ids.append(doc_id)
                embeddings_list.append(embedding)
                metadatas.append({
                    "source": source,
                    "title": title,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                })
                texts.append(chunk)

            if ids:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=texts,
                )
                chunks_added += len(ids)
                logger.debug(f"Ingested {len(ids)} chunks from: {source}")

        logger.info(f"Total chunks ingested this session: {chunks_added}")
        return chunks_added

    # ── Retrieval ─────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Retrieve the most relevant chunks for a query.
        Returns list of {"text", "source", "title", "score"}
        """
        if self.collection.count() == 0:
            logger.warning("ChromaDB collection is empty — no context to retrieve.")
            return []

        query_embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()

        where = {"source": source_filter} if source_filter else None

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "text": doc,
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "score": round(1 - dist, 4),  # cosine similarity
            })

        return retrieved

    def retrieve_as_context(self, query: str, top_k: int = TOP_K_RESULTS) -> str:
        """Return retrieved chunks as a single formatted context string."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant context available."

        context_parts = []
        for i, r in enumerate(results, 1):
            source = r.get("source", "")
            title = r.get("title", "")
            header = f"[Source {i}: {title or source}]"
            context_parts.append(f"{header}\n{r['text']}")

        return "\n\n---\n\n".join(context_parts)

    # ── Management ────────────────────────────────────────────

    def reset_collection(self):
        """Clear all documents from the collection."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{self.collection_name}' has been reset.")

    def count(self) -> int:
        return self.collection.count()


# ── Singleton ─────────────────────────────────────────────────

_rag_instance: Optional[RAGPipeline] = None


def get_rag_pipeline(collection_name: str = CHROMA_COLLECTION_NAME) -> RAGPipeline:
    global _rag_instance
    if _rag_instance is None or _rag_instance.collection_name != collection_name:
        _rag_instance = RAGPipeline(collection_name=collection_name)
    return _rag_instance
