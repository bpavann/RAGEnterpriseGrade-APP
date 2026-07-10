import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 10
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_gemini_model = None
_mpnet_model = None
_active_model = None
_model_type = None   # "gemini" or "fallback"

def _probe_gemini():
    """
    Verify Gemini embedding API availability.
    Only used for testing/fallback demonstration.
    """
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        # Test request
        model.embed_query("health check")
        logfire.info("Gemini embedding active | model=gemini-embedding-2-preview | dim=3072")
        return model
    except Exception as e:
        logfire.warning(f"Gemini embedding unavailable: {e}")
        return None

def _load_fallback():
    """
    Load local sentence-transformer model.
    """
    logfire.info(
        "Loading fallback embedding model | "
        "model=all-mpnet-base-v2 | dim=768")
    return SentenceTransformer("all-mpnet-base-v2")

def _init():

    global _gemini_model
    global _mpnet_model
    global _active_model
    global _model_type
    if _active_model is not None:
        return
    # Test Gemini only for monitoring/demo
    _gemini_model = _probe_gemini()
    # Always use local embedding for Qdrant
    _mpnet_model = _load_fallback()
    _active_model = _mpnet_model
    _model_type = "fallback"
    if _gemini_model:
        logfire.info(
            "Gemini available but using local embedding "
            "for consistent Qdrant vectors")
    else:
        logfire.info(
            "Gemini failed. "
            "Using local fallback embedding model")

# ── Public helpers ─────────────────────────────────────────────────────────────
def get_embedding_dim() -> int:
    _init()
    return _FALLBACK_DIM

def get_embedding_model_type() -> str:
    """
    Used for logging/debugging.
    """
    _init()
    return _model_type

# ── Batch embedding with retry ─────────────────────────────────────────────────    
def _embed_batch(batch: list[str]) -> list[list[float]]:
    _init()
    return _active_model.encode(batch,show_progress_bar=False).tolist()  

# ── Public API (same signatures as before) ─────────────────────────────────────
def embed_query(query: str) -> list[float]:
    _init()
    return _active_model.encode([query],show_progress_bar=False)[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    embeddings = []
    for i in range(0,len(texts),BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        with logfire.span("Embedding batch",model=_model_type,start=i,size=len(batch)):
            embeddings.extend(_embed_batch(batch))
    return embeddings