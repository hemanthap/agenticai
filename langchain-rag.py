"""
Simple RAG (Retrieval-Augmented Generation) using LangChain + Qdrant
=====================================================================
Uses the modern LCEL (LangChain Expression Language) pipeline —
no deprecated langchain.chains or langchain.schema imports.

Requirements:
    pip install langchain langchain-openai langchain-qdrant \
                langchain-text-splitters qdrant-client tiktoken

Usage:
    export OPENAI_API_KEY="sk-..."
    python simple_rag.py
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
def split_documents(docs: list[Document], chunk_size: int = 300, chunk_overlap: int = 50):
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
QDRANT_URL        = "http://qdrant.home"
QDRANT_COLLECTION = "rag_demo"
EMBEDDING_DIM     = 1536   # text-embedding-ada-002 / text-embedding-3-small


def build_vector_store(chunks: list[Document]) -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings()

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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
    answer = chain.invoke(question)
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
    openai_api_key = os.getenv('OPENAI_API_KEY')
    # if not os.getenv("openai_api_key"):
    #     raise EnvironmentError("Set OPENAI_API_KEY before running.")

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