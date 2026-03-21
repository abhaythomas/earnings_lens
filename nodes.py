"""
nodes.py — LangGraph Node Functions for EarningsLens

Each function here is a "node" in our agentic RAG graph.
A node takes the current state, does something, and returns updated state.

The nodes are:
1. route_question    → Decides: does this need retrieval or not?
2. retrieve          → Fetches relevant chunks from the vector store
3. grade_documents   → Checks if retrieved chunks are actually relevant
4. generate          → Produces an answer from relevant chunks
5. check_hallucination → Verifies the answer is grounded in the documents
6. check_usefulness  → Verifies the answer actually addresses the question
7. rewrite_query     → Rewrites the question if retrieval failed
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"  # Free on Groq, very capable
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Initialize shared resources ─────────────────────────────────────
llm = ChatGroq(model=GROQ_MODEL, temperature=0)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},  # Retrieve top 5 most relevant chunks
)


# ── Node 1: Route Question ──────────────────────────────────────────
def route_question(state: dict) -> dict:
    """
    Decides whether the question needs document retrieval or can be
    answered directly. For an earnings call app, almost everything
    needs retrieval — but general questions like "what is revenue?"
    don't need the vector store.

    Returns: state with "route" set to "retrieve" or "direct"
    """
    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a router for an earnings call analysis system.
Decide if the question requires searching earnings call transcripts or can be answered from general knowledge.

Answer ONLY with one word:
- "retrieve" if the question is about a specific company, earnings, financials, or something said in a call
- "direct" if it's a general knowledge question (e.g., "what does EPS mean?")"""),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().lower()

    route = "retrieve" if "retrieve" in result else "direct"
    print(f"🔀 Router: {route}")
    return {**state, "route": route}


# ── Node 2: Retrieve Documents ───────────────────────────────────────
def retrieve(state: dict) -> dict:
    """
    Fetches the top-k most similar document chunks from ChromaDB.
    Uses cosine similarity between the question embedding and stored chunk embeddings.
    """
    question = state["question"]
    print(f"🔍 Retrieving documents for: {question[:80]}...")

    documents = retriever.invoke(question)
    print(f"   Found {len(documents)} chunks")

    return {**state, "documents": documents}


# ── Node 3: Grade Documents ──────────────────────────────────────────
def grade_documents(state: dict) -> dict:
    """
    Filters retrieved documents — keeps only those that are actually
    relevant to the question. Uses a SINGLE LLM call to grade all
    documents at once (instead of one call per document) to minimize
    API usage and avoid rate limits.

    Returns: state with "documents" filtered and "documents_relevant" flag
    """
    question = state["question"]
    documents = state["documents"]

    # Format all documents with indices for batch grading
    docs_text = "\n\n".join(
        f"[Document {i+1}]\n{doc.page_content[:500]}"
        for i, doc in enumerate(documents)
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a relevance grader for an earnings call analysis system.
Given a user question and multiple retrieved document chunks, determine which chunks contain information relevant to answering the question.

Reply with ONLY the numbers of relevant documents, comma-separated. Example: "1, 3, 5"
If none are relevant, reply with "none"."""),
        ("human", "Question: {question}\n\nDocuments:\n{documents}"),
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "documents": docs_text}).strip().lower()

    relevant_docs = []
    if "none" not in result:
        for i, doc in enumerate(documents):
            if str(i + 1) in result:
                relevant_docs.append(doc)

    print(f"📋 Grader: {len(relevant_docs)}/{len(documents)} chunks are relevant")

    documents_relevant = len(relevant_docs) > 0
    return {**state, "documents": relevant_docs, "documents_relevant": documents_relevant}


# ── Node 4: Generate Answer ──────────────────────────────────────────
def generate(state: dict) -> dict:
    """
    Generates an answer using ONLY the relevant document chunks as context.
    The prompt explicitly instructs the LLM to not make up information
    and to cite which source each piece of information comes from.
    """
    question = state["question"]
    documents = state["documents"]

    # Format documents with source metadata for citation
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in documents
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial analyst assistant. Answer the question based ONLY on the provided earnings call transcript excerpts.

Rules:
- Only use information from the provided context
- Cite the source file for each claim (e.g., "According to apple_q4_2024.txt...")
- If the context doesn't contain enough information, say so clearly
- Be concise and specific — use numbers and quotes when available
- Structure your answer with clear points"""),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # Track how many generation attempts we've made (for loop limiting)
    generation_count = state.get("generation_count", 0) + 1
    print(f"💡 Generated answer (attempt {generation_count})")

    return {**state, "answer": answer, "generation_count": generation_count}


# ── Node 5: Direct Answer (no retrieval needed) ─────────────────────
def direct_answer(state: dict) -> dict:
    """
    For general knowledge questions that don't need the vector store.
    """
    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful financial analyst assistant.
Answer the question from your general knowledge. Be concise."""),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question})

    return {**state, "answer": answer, "route": "direct"}


# ── Node 6: Check Hallucination ──────────────────────────────────────
def check_hallucination(state: dict) -> dict:
    """
    Verifies that the generated answer is actually grounded in the
    retrieved documents. This is critical for financial data — you
    don't want the model making up revenue numbers.

    Returns: state with "is_grounded" flag
    """
    documents = state["documents"]
    answer = state["answer"]

    context = "\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a fact-checker for financial information.
Given source documents and a generated answer, determine if the answer is fully supported by the documents.

Answer ONLY "yes" if the answer is grounded in the documents, or "no" if it contains claims not supported by the sources."""),
        ("human", "Documents:\n{context}\n\nAnswer:\n{answer}"),
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "answer": answer}).strip().lower()

    is_grounded = "yes" in result
    print(f"🔎 Hallucination check: {'grounded ✅' if is_grounded else 'not grounded ❌'}")

    return {**state, "is_grounded": is_grounded}


# ── Node 7: Check Usefulness ─────────────────────────────────────────
def check_usefulness(state: dict) -> dict:
    """
    Checks if the answer actually addresses the user's question.
    An answer can be grounded (not hallucinated) but still not useful
    if it doesn't answer what was asked.

    Returns: state with "is_useful" flag
    """
    question = state["question"]
    answer = state["answer"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an answer quality evaluator.
Given a question and an answer, determine if the answer actually addresses the question.

Answer ONLY "yes" if the answer is useful and addresses the question, or "no" if it doesn't."""),
        ("human", "Question: {question}\n\nAnswer: {answer}"),
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "answer": answer}).strip().lower()

    is_useful = "yes" in result
    print(f"✅ Usefulness check: {'useful ✅' if is_useful else 'not useful ❌'}")

    return {**state, "is_useful": is_useful}


# ── Node 8: Rewrite Query ────────────────────────────────────────────
def rewrite_query(state: dict) -> dict:
    """
    If retrieval failed (no relevant documents found) or the answer
    wasn't useful, we rewrite the query to try a different angle.
    This is the "self-correcting" part of Adaptive RAG.
    """
    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query rewriter for an earnings call search system.
The original query didn't return good results. Rewrite it to be more specific or use different terminology that might match earnings call language.

Return ONLY the rewritten query, nothing else."""),
        ("human", "Original query: {question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    new_question = chain.invoke({"question": question}).strip()

    print(f"✏️  Query rewritten: {new_question[:80]}...")

    return {**state, "question": new_question}
