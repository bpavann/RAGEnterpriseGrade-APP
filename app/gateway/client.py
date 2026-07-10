import logfire
from portkey_ai import Portkey,createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI
from app.config import settings

logfire.configure()

# Initialize Portkey client
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=settings.PORTKEY_CONFIG_ID
)

# Function to create a Portkey-backed ChatOpenAI client
def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI client.

    Portkey exposes an OpenAI-compatible endpoint, allowing requests to be
    routed through the configured gateway. The routing, fallback, retry,
    and cache behavior are controlled by the Portkey Config referenced by
    PORTKEY_CONFIG_ID.
    """
    with logfire.span("llm_creation", feature=feature):
       
        return ChatOpenAI(
            base_url=PORTKEY_GATEWAY_URL,
            api_key=settings.PORTKEY_API_KEY,
            # Primary model.
            # Fallbacks are handled by the Portkey Config.
            model=f"@{settings.OPENAI_SLUG}/gpt-4.1-mini",
            temperature=0,
            default_headers=createHeaders(
                api_key=settings.PORTKEY_API_KEY,
                config=settings.PORTKEY_CONFIG_ID,
                metadata={
                    "feature": feature,
                    "_user": "rag-system",
                    "environment": "production",
                },
            ),
        )

# Function to extract cache status from Portkey response
def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey response headers.
    Returns "MISS" if the header is unavailable.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"