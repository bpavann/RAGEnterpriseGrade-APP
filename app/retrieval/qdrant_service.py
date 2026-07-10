import logfire
from qdrant_client import QdrantClient
from app.config import settings
from app.retrieval.embeddings import embed_query

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

# Search Function
def search_enterprise_knowledge(query: str,limit: int = 15) -> list[dict]:
    """
    Search enterprise knowledge from the
    correct Qdrant collection.
    """
    if not query:
        return []
    try:
        # Generate query embedding
        vector = embed_query(query)
        # Select matching collection
        collection_name = settings.QDRANT_COLLECTION
        logfire.info("Qdrant search",collection=collection_name,embedding_model="all-mpnet-base-v2",limit=limit)
        hits = qdrant_client.query_points(collection_name=collection_name,query=vector,limit=limit,with_payload=True,).points
    except Exception as exc:
        logfire.error(f"Qdrant search failed: {exc}")
        raise

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            {
                "content":payload.get("text") or payload.get("content") or "",
                "source":payload.get("source"),
                "score":hit.score,
            }
        )
    return results
