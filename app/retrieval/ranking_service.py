import logging
import numpy as np
from app.retrieval.embeddings import embed_query, embed_texts


def _cosine_similarity(query_vec: list[float], doc_vecs: list[list[float]]) -> list[float]:
    query_array = np.asarray(query_vec, dtype=np.float32)
    doc_arrays = np.asarray(doc_vecs, dtype=np.float32)
    if doc_arrays.size == 0:
        return []

    query_norm = np.linalg.norm(query_array)
    doc_norms = np.linalg.norm(doc_arrays, axis=1)
    denom = query_norm * doc_norms
    denom = np.where(denom == 0, 1e-12, denom)
    return np.dot(doc_arrays, query_array) / denom


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    if not documents:
        return []
    if len(documents) <= top_n:
        return documents

    try:
        query_vector = embed_query(query)
        doc_vectors = embed_texts(documents)
        scores = _cosine_similarity(query_vector, doc_vectors)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [documents[i] for i in ranked_indices[:top_n]]
    except Exception as exc:
        logging.getLogger(__name__).warning(
            f"Reranking failed, returning top {top_n} original docs: {exc}"
        )
        return documents[:top_n]
