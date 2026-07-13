import time 
import logfire
import numpy as np
#from app.retrieval.embeddings import embed_query, embed_texts
from flashrank import Ranker, RerankRequest

_ranker=None

# Reranking Function using FlashRank method============================================
def _get_ranker()-> Ranker:
    """
    Initializes the FlashRank engine lazily. 
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing FlashRank reranker with ms-marco-MiniLM-L-6-v2 model")
        try:
            _ranker=Ranker(cache_dir="temp/flashrank")
        except Exception:
            _ranker=Ranker()
    return _ranker

def rerank_documents_fr(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    
    Why FlashRank? 
    Standard vector search (Cosine Similarity) is fast but mathematically "fuzzy."
    FlashRank uses a Cross-Encoder approach which is much more precise but usually slow.
    FlashRank solves this by using highly optimized, quantized ONNX models locally.
    """
    if not documents:
        return []
    
    start_time=time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")
    try:
        ranker=_get_ranker()
        passages=[
            {"id":i, "text":doc} for i,doc in enumerate(documents)
        ]
        request=RerankRequest(query=query,passages=passages)
        result=ranker.rerank(request)

        reranked_doc=[]
        for rr in result[:top_n]:
            reranked_doc.append(rr['text'])
        
        duration=time.time()-start_time
        top_score=result[0]["score"] if result else None
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}")
        return reranked_doc
    
    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fallback to the original Qdrant order to ensure the user still gets an answer
        return documents[:top_n]


"""
# Reranking Function Cosine Similarity method ============================================================
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

def rerank_documents_cs(query: str, documents: list[str], top_n: int = 5) -> list[str]:

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
    
    except Exception as e:
        logfire.error(f"❌ [Reranker] Cosine Similarity Reranking Failed: {e}")
        return documents[:top_n]
"""