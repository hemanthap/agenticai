#!/usr/bin/env python3

# pip install langchain-anthropic langchain-community langchain-text-splitters chromadb sentence-transformers
# pip install -U langchain-huggingface langchain-chroma --break-system-packages

import os
import chromadb
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Initialize Claude LLM
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    api_key='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',
)

# Initialize embeddings (updated package)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Chroma HTTP client — correct way to connect to a remote Chroma instance
chroma_http_client = chromadb.HttpClient(host="chroma.home", port=80)

chroma_client = Chroma(
    client=chroma_http_client,
    collection_name="documents",
    embedding_function=embeddings,
)

# Sample documents
documents = [
    Document(
        page_content="Python is a high-level, interpreted programming language known for its simplicity and readability.",
        metadata={"source": "intro", "topic": "Python"}
    ),
    Document(
        page_content="Machine learning involves training algorithms to learn patterns from data without explicit programming.",
        metadata={"source": "ml_intro", "topic": "Machine Learning"}
    ),
    Document(
        page_content="Vector databases store and retrieve high-dimensional vectors efficiently using similarity search.",
        metadata={"source": "vector_db", "topic": "Databases"}
    ),
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_intro", "topic": "LangChain"}
    ),
    Document(
        page_content="Chroma is an open-source vector database designed for AI applications and semantic search.",
        metadata={"source": "chroma_intro", "topic": "Vector Databases"}
    ),
]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
)

all_chunks = []
for doc in documents:
    chunks = text_splitter.split_text(doc.page_content)
    for chunk in chunks:
        all_chunks.append(
            Document(page_content=chunk, metadata=doc.metadata)
        )

print(f"Total documents: {len(documents)}")
print(f"Total chunks after splitting: {len(all_chunks)}")

# Store chunks in Chroma
try:
    chroma_client.add_documents(all_chunks)
    print(f"✓ Successfully stored {len(all_chunks)} chunks in Chroma")
except Exception as e:
    print(f"✗ Error storing chunks: {e}")
    exit(1)

# Query
print("\n--- Testing retrieval ---")
query = "Tell me about vector databases"
results = chroma_client.similarity_search(query, k=3)
print(f"\nQuery: '{query}'")
print(f"Found {len(results)} similar documents:\n")
for i, result in enumerate(results, 1):
    print(f"{i}. {result.page_content[:100]}...")
    print(f"   Metadata: {result.metadata}\n")

# Use Claude with retrieved context
print("\n--- Using Claude with retrieved context ---")
context = "\n".join([doc.page_content for doc in results])
prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}
Answer:"""

# response = llm.invoke(prompt)
# print(f"Claude's response:\n{response.content}")

print("--- Skipping Claude call (no credits) ---")
print(f"Prompt that would be sent:\n{prompt}")