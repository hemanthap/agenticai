"""
Simple RAG (Retrieval-Augmented Generation) using LangChain + Qdrant
=====================================================================
Uses the modern LCEL (LangChain Expression Language) pipeline —
no deprecated langchain.chains or langchain.schema imports.

Requirements:
    pip install langchain langchain-mistralai langchain-qdrant \
                langchain-text-splitters qdrant-client --break-system-packages

Usage:
    export MISTRAL_API_KEY="..."  # used for embeddings + chat model
    export MISTRAL_CHAT_MODEL="ministral-8b-latest"  # optional, higher-throughput default
    python simple_rag.py
"""

import os
import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# ---------------------------------------------------------------------------
# 1. Sample documents (swap in PyPDFLoader, TextLoader, etc. for real files)
# ---------------------------------------------------------------------------
SAMPLE_DOCS = [
    Document(
        page_content=(
            "LangChain is a framework for building applications powered by large "
            "language models. It provides abstractions for chains, agents, memory, "
            "and retrieval, making it easy to connect LLMs to external data sources."
        ),
        metadata={"source": "langchain_overview.txt"},
    ),
    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) combines a retrieval step with "
            "text generation. A query is first used to fetch relevant documents from "
            "a knowledge base, and those documents are then passed to an LLM so it "
            "can produce a grounded, factual answer."
        ),
        metadata={"source": "rag_intro.txt"},
    ),
    Document(
        page_content=(
            "Qdrant is a high-performance vector database designed for production "
            "semantic search. It stores vectors alongside JSON payloads and supports "
            "filtering, named collections, and both in-memory and persistent storage."
        ),
        metadata={"source": "qdrant_overview.txt"},
    ),
    Document(
        page_content=(
            "Text splitters break long documents into smaller, overlapping chunks "
            "before embedding them. RecursiveCharacterTextSplitter tries to split on "
            "paragraph boundaries first, then sentences, then words, preserving "
            "semantic coherence as much as possible."
        ),
        metadata={"source": "text_splitting.txt"},
    ),
    Document(
        page_content=(
            "OpenAI Embeddings convert text into high-dimensional vectors. Similar "
            "pieces of text produce vectors that are close together in vector space, "
            "which is the foundation of semantic search in RAG pipelines."
        ),
        metadata={"source": "embeddings.txt"},
    ),
]


# ---------------------------------------------------------------------------
# 2. Split documents into chunks
# ---------------------------------------------------------------------------
def split_documents(docs: list[Document], chunk_size: int = 300, chunk_overlap: int = 50)->list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"[split]    {len(docs)} doc(s) -> {len(chunks)} chunk(s)")
    return chunks


# ---------------------------------------------------------------------------
# 3. Build a Qdrant vector store
# ---------------------------------------------------------------------------
QDRANT_URL        = "http://192.168.68.50:6333"   # adjust if your Qdrant is elsewhere
QDRANT_COLLECTION = "rag_demo"
EMBEDDING_MODEL   = "mistral-embed"
EMBEDDING_DIM     = 1024   # mistral-embed
CHAT_MODEL        = os.getenv("MISTRAL_CHAT_MODEL", "ministral-8b-latest")


def build_vector_store(chunks: list[Document]) -> QdrantVectorStore:
    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)

    client = QdrantClient(url=QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)
        print(f"[qdrant]   Dropped existing collection '{QDRANT_COLLECTION}'")

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"[qdrant]   Created collection '{QDRANT_COLLECTION}' at {QDRANT_URL}")

    store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
    )
    print(f"[index]    Indexed {len(chunks)} chunk(s) into Qdrant")
    return store


# ---------------------------------------------------------------------------
# 4. Build the RAG chain (LCEL style)
# ---------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use ONLY the context below to answer \
the question. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store: QdrantVectorStore):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatMistralAI(model=CHAT_MODEL, temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return retriever, chain


# ---------------------------------------------------------------------------
# 5. Query helper
# ---------------------------------------------------------------------------
def ask(retriever, chain, question: str):
    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    answer = None
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            answer = chain.invoke(question)
            break
        except Exception as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code == 429 and attempt < max_retries:
                wait_seconds = 2 ** attempt
                print(f"   [warn] Rate-limited by Mistral (429). Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue
            if status_code == 429:
                answer = "I don't know. (Mistral API rate limit exceeded. Please retry shortly.)"
                break
            raise

    if answer is None:
        answer = "I don't know."

    print(f"A: {answer}")
    docs = retriever.invoke(question)
    sources = {doc.metadata.get("source", "unknown") for doc in docs}
    print(f"   Sources: {', '.join(sorted(sources))}")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv(override=True)
    if not os.getenv("MISTRAL_API_KEY"):
        raise EnvironmentError("Set MISTRAL_API_KEY before running.")
    print(f"[model]    Chat model: {CHAT_MODEL}")

    chunks = split_documents(SAMPLE_DOCS)
    store = build_vector_store(chunks)
    retriever, chain = build_rag_chain(store)

    questions = [
        "What is LangChain?",
        "How does RAG work?",
        "What is Qdrant used for?",
        "Why do we split documents into chunks?",
        "What is the capital of Mars?",   # out-of-context -> "I don't know"
    ]

    for q in questions:
        ask(retriever, chain, q)