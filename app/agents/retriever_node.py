import logfire
from app.agents.state import AgentState
from app.retrieval.qdrant_service import search_enterprise_knowledge
from app.retrieval.ranking_service import rerank_documents_fr

def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]
    
    
    # Standard Retrieval Logic which uses Qdrant for vector search and Standard Cosine Similarity for reranking
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")
        
        doc_contents = [doc['content'] for doc in raw_results]
        
        with logfire.span("⚖️ Semantic Reranking"):
            # Rerank the retrieved documents using Cosine Similarity
            #reranked_contents_cs = rerank_documents_cs(query, doc_contents, top_n=5)

            # Rerank the retrieved documents using FlashRank Cross-Encoder
            reranked_contents_fr = rerank_documents_fr(query, doc_contents, top_n=5)

            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")
            
        #formatted_docs_cs = [f"CONTENT: {doc}" for doc in reranked_contents_cs]
        formatted_docs_fr = [f"CONTENT: {doc}" for doc in reranked_contents_fr]
    
    return {
        "documents": formatted_docs_fr,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }