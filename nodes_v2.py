"""
nodes_v2.py — Node Functions for EarningsLens v2

What changed from nodes.py:
- get_vectorstore() loads from Pinecone (cloud) instead of ChromaDB (local)
- get_loaded_companies_v2() queries Pinecone metadata instead of local collection
- No module-level vectorstore/retriever initialization — lazy-loaded + cached
- compare_companies and retrieve use get_vectorstore() instead of globals

Everything else — all LLM prompts, grading logic, hallucination check,
usefulness check, rewrite logic — is IDENTICAL to v1.
"""

import os
import sys
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PINECONE_INDEX_NAME = "earningslens-v2"   # Must match ingest_v2.py

# ── LLM (same as v1) ────────────────────────────────────────────────
llm = ChatGroq(model=GROQ_MODEL, temperature=0)

# ── Cached Pinecone vectorstore ──────────────────────────────────────
_vectorstore_cache = None

def get_vectorstore() -> PineconeVectorStore:
    """
    Load the Pinecone vector store with HuggingFace embeddings.

    Cached after first load — Streamlit reruns this module on every
    interaction, so without caching we'd re-initialize the embedding
    model on every query (slow).

    This replaces the v1 pattern:
        Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    With:
        PineconeVectorStore(index_name=..., embedding=embeddings)

    The returned object has the same .as_retriever() interface,
    so all downstream nodes work unchanged.
    """
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    _vectorstore_cache = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
    return _vectorstore_cache


def get_loaded_companies_v2() -> list[str]:
    """
    Return the list of unique source filenames in the Pinecone index.

    In v1 this queried ChromaDB's local metadata collection directly.
    In v2 we do a broad similarity search and deduplicate the 'source'
    metadata field from the results.

    This is a lightweight probe — we fetch 100 vectors and deduplicate.
    For larger indexes, Pinecone's metadata filtering API can be used.
    """
    try:
        vs = get_vectorstore()
        results = vs.similarity_search("earnings revenue", k=100)
        sources = list({doc.metadata.get("source", "") for doc in results if doc.metadata.get("source")})
        return sources
    except Exception:
        return []


# ── Node 1: Route Question ──────────────────────────────────────────
def route_question(state: dict) -> dict:
    """
    Decides the query type:
    - "compare" if the user wants to compare multiple companies
    - "retrieve" if the question is about specific company data
    - "direct" if it's a general knowledge question
    """
    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a router for an earnings call analysis system.
Decide the type of query:

Answer ONLY with one word:
- "compare" if the user wants to compare two or more companies (e.g., "compare Apple and Microsoft", "how does X differ from Y", "X vs Y")
- "retrieve" if the question is about a specific company's earnings, financials, or something said in a call
- "direct" if it's a general knowledge question (e.g., "what does EPS mean?")"""),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().lower()

    if "compare" in result:
        route = "compare"
    elif "retrieve" in result:
        route = "retrieve"
    else:
        route = "direct"

    print(f"🔀 Router: {route}")
    return {**state, "route": route}


# ── Node 9: Compare Companies ────────────────────────────────────────
def compare_companies(state: dict) -> dict:
    """
    Handles comparison queries by:
    1. Extracting company names from the question
    2. Retrieving documents for EACH company separately from Pinecone
    3. Generating a structured side-by-side comparison

    This gives much better results than a single retrieval because
    it ensures both companies get equal representation in the context.
    """
    question = state["question"]
    print(f"⚖️  Compare Mode: {question[:80]}...")

    # Step 1: Extract company names from the question
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract the company names being compared from this question.
Return ONLY the company names, comma-separated. Example: "Apple, Microsoft"
If you can't identify specific companies, return "unknown"."""),
        ("human", "{question}"),
    ])

    chain = extract_prompt | llm | StrOutputParser()
    companies_str = chain.invoke({"question": question}).strip()
    companies = [c.strip() for c in companies_str.split(",") if c.strip()]
    print(f"   Companies identified: {companies}")

    # Step 2: Retrieve documents for each company separately from Pinecone
    vs = get_vectorstore()
    all_docs = []
    company_docs = {}

    for company in companies:
        docs = vs.similarity_search(
            f"{company} {question}",
            k=5,
        )
        relevant = [d for d in docs if company.lower() in d.page_content.lower()
                    or company.lower() in d.metadata.get("source", "").lower()]

        if not relevant:
            relevant = docs[:3]

        company_docs[company] = relevant
        all_docs.extend(relevant)
        print(f"   {company}: found {len(relevant)} relevant chunks")

    # Step 3: Generate structured comparison
    context_parts = []
    for company, docs in company_docs.items():
        company_context = "\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )
        context_parts.append(f"=== {company.upper()} ===\n{company_context}")

    full_context = "\n\n".join(context_parts)

    compare_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial analyst. Compare the companies based ONLY on the provided earnings call excerpts.

Rules:
- Use ONLY information from the provided context
- Structure your comparison with clear categories (Revenue, Growth, Strategy, Risks, Outlook)
- Use a side-by-side format where possible
- Cite the source file for each claim
- If information is missing for one company, say so explicitly
- Use specific numbers and quotes when available
- End with a brief summary of key differences"""),
        ("human", "Context:\n{context}\n\nComparison question: {question}"),
    ])

    chain = compare_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": full_context, "question": question})

    generation_count = state.get("generation_count", 0) + 1
    print(f"💡 Comparison generated")

    return {
        **state,
        "answer": answer,
        "documents": all_docs,
        "route": "compare",
        "generation_count": generation_count,
        "companies_compared": companies,
    }


# ── Node 2: Retrieve Documents ───────────────────────────────────────
def retrieve(state: dict) -> dict:
    """
    Fetches the top-k most similar document chunks from Pinecone.
    Uses cosine similarity between the question embedding and stored chunk embeddings.
    """
    question = state["question"]
    print(f"🔍 Retrieving documents for: {question[:80]}...")

    retriever = get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    documents = retriever.invoke(question)
    print(f"   Found {len(documents)} chunks")

    return {**state, "documents": documents}


# ── Node 3: Grade Documents ──────────────────────────────────────────
def grade_documents(state: dict) -> dict:
    """
    Filters retrieved documents — keeps only those that are actually
    relevant to the question. Uses a SINGLE LLM call to grade all
    documents at once to minimize API usage and avoid rate limits.
    """
    question = state["question"]
    documents = state["documents"]

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
    The prompt explicitly instructs the LLM not to make up information
    and to cite which source each piece of information comes from.
    """
    question = state["question"]
    documents = state["documents"]

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
    Verifies that the generated answer is grounded in the retrieved
    documents. Critical for financial data — we don't want the model
    fabricating revenue numbers.
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
    If retrieval failed or the answer wasn't useful, rewrites the query
    to try a different angle. This is the self-correcting part of Adaptive RAG.
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
