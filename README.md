# **RAG Enterprise-Grade Application**


**This project is a **production-grade, scalable, and advanced Retrieval-Augmented Generation (RAG) application** designed to handle high-noise enterprise data. Moving beyond simple prototypes, this system utilizes an **Agentic AI** architecture to intelligently route queries, ensure robust security, and provide deep observability into the AI's execution.**


## Core Features

### 1. Agentic Orchestration
Instead of a fixed pipeline, the system uses **LangGraph** to manage a dynamic workflow between specialized nodes:
*   **Planner Node:** Analyzes the user's intent to decide if a query is conversational or requires technical retrieval.
*   **Retriever Node:** Fetches relevant data from the vector database and applies **semantic reranking** to prioritize accuracy.
*   **Responder Node:** Synthesizes final answers using retrieved context and session memory.

### 2. Advanced Data Ingestion & Retrieval
*   **Smart Parsing:** Handles multiple formats including PDF, HTML, Docx, and PPTX using specialized loaders.
*   **Vector Storage:** Uses **Qdrant Cloud** for high-performance semantic search.
*   **Hybrid Reranking:** Employs **FlashRank** (locally) or **Jina** (production) to refine search results and eliminate noise.
*   **Embedding Strategy:** Utilizes **Google Gemini** as the primary model with **Sentence Transformers** as a robust local fallback.

### 3. Enterprise Security & Guardrails
Implemented via **NVIDIA NeMo Guardrails** to protect the integrity of the system:
*   **Input Guardrails:** Detects and blocks prompt injections, off-topic queries, and PII (Personally Identifiable Information).
*   **Output Guardrails:** Validates responses to prevent hallucinations or the leakage of sensitive data.

### 4. LLM Gateway & Resilience
Managed through **Portkey** to ensure 100% uptime and cost optimization:
*   **Model Routing & Fallbacks:** Automatically switches to a backup model (e.g., switching from OpenAI to Gemini) if primary API limits are hit.
*   **Semantic Caching:** Reduces costs and latency by storing and reusing answers for semantically similar queries.

### 5. Deep Observability
Complete transparency into system performance and AI thought processes:
*   **Pydantic Logfire:** Provides detailed application tracing, allowing developers to monitor every "span" and "trace" in the execution flow.
*   **LangSmith:** Used for tracing large language model execution and debugging agentic behavior.

## Technical Stack
*   **Frontend:** Streamlit.
*   **Backend:** FastAPI & LangGraph.
*   **Vector Database:** Qdrant Cloud.
*   **LLMs:** Google Gemini, Groq (Llama-3), and Sentence Transformers.
*   **Infrastructure:** AWS with automated CI/CD pipelines.