"""
eval_ragas.py — RAGAS Evaluation for EarningsLens v2

Two modes:
  --generate   Pull chunks from Pinecone, use Groq to create synthetic Q&A pairs,
               save to eval/eval_dataset.json
  --run        Load eval_dataset.json, run each question through the RAG pipeline,
               score with RAGAS, save results to eval/scores_<date>.json

Usage:
  # Step 1: generate the dataset (only needed once, or when index changes)
  python eval_ragas.py --generate --n 40

  # Step 2: run the eval
  python eval_ragas.py --run

  # Both in one shot:
  python eval_ragas.py --generate --run --n 40
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime, timezone
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

EVAL_DIR = "eval"
DATASET_FILE = os.path.join(EVAL_DIR, "eval_dataset.json")
PIPELINE_RESULTS_FILE = os.path.join(EVAL_DIR, "pipeline_results.json")
os.makedirs(EVAL_DIR, exist_ok=True)

# ── Seed queries: diverse financial topics to pull varied chunks ──────
# These are used to sample different parts of the index, not as eval questions.
SEED_QUERIES = [
    "revenue growth quarterly earnings",
    "operating margin cost reduction",
    "AI artificial intelligence investment strategy",
    "supply chain disruption risk",
    "guidance outlook next quarter forecast",
    "free cash flow capital allocation",
    "market share competitive landscape",
    "CEO comments strategic priorities",
    "international revenue foreign exchange",
    "gross margin product mix",
    "research development spending innovation",
    "shareholder return buyback dividend",
    "debt balance sheet liquidity",
    "headcount hiring workforce",
    "cloud software subscription recurring revenue",
    "semiconductor chip demand inventory",
    "consumer spending retail sales",
    "interest rate inflation impact",
    "acquisition merger integration",
    "regulatory risk compliance",
]


# ── Step 1: Generate synthetic dataset ───────────────────────────────

def generate_dataset(n: int = 40) -> list[dict]:
    """
    Pull chunks from Pinecone via diverse seed queries, then use Groq to
    generate one (question, ground_truth) pair per chunk.

    Each row:
    {
        "question": str,
        "ground_truth": str,       # reference answer
        "reference_contexts": [str] # the chunk used to generate the Q&A
    }
    """
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from nodes_v2 import get_vectorstore

    print(f"\n{'='*60}")
    print(f"  Generating synthetic eval dataset  (target: {n} rows)")
    print(f"{'='*60}\n")

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
    vs = get_vectorstore()

    # Sample chunks via diverse seed queries, deduplicate by content hash
    seen_hashes: set[str] = set()
    candidate_chunks = []

    queries_to_use = SEED_QUERIES * ((n // len(SEED_QUERIES)) + 2)
    random.shuffle(queries_to_use)

    for seed in queries_to_use:
        if len(candidate_chunks) >= n * 3:  # over-sample, we'll trim later
            break
        try:
            docs = vs.similarity_search(seed, k=5)
            for doc in docs:
                h = str(hash(doc.page_content[:200]))
                if h not in seen_hashes and len(doc.page_content.strip()) >= 300:
                    seen_hashes.add(h)
                    candidate_chunks.append(doc)
        except Exception as e:
            print(f"  ⚠ Seed query failed: {e}")
            continue

    random.shuffle(candidate_chunks)
    candidate_chunks = candidate_chunks[:n * 2]  # keep 2x buffer for failures

    print(f"  Sampled {len(candidate_chunks)} unique chunks from Pinecone\n")

    # Generate Q&A pairs from each chunk
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are building an evaluation dataset for a financial RAG system.

Given an earnings call excerpt, generate:
1. A specific, factual question that can ONLY be answered using this excerpt
2. A concise, accurate answer using only information from the excerpt

Rules:
- The question must require a specific number, statement, or fact from the text
- Do NOT ask vague questions like "what did the CEO say?" — be specific
- The answer must be 1-3 sentences, citing the specific figure or quote
- Do not reference "the excerpt" or "the document" in the question — phrase it naturally

Return ONLY valid JSON in this exact format:
{{
  "question": "...",
  "answer": "..."
}}"""),
        ("human", "Source: {source}\n\nExcerpt:\n{text}"),
    ])

    chain = qa_prompt | llm | StrOutputParser()

    dataset = []
    failed = 0

    for i, doc in enumerate(candidate_chunks):
        if len(dataset) >= n:
            break

        source = doc.metadata.get("source", "unknown")
        text = doc.page_content[:1200]  # cap to avoid token overflow

        print(f"  [{i+1}/{len(candidate_chunks)}] Generating Q&A from {source[:50]}...", end="", flush=True)

        try:
            raw = chain.invoke({"source": source, "text": text}).strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            question = parsed.get("question", "").strip()
            answer = parsed.get("answer", "").strip()

            if not question or not answer or len(question) < 20:
                print(" ✗ (bad output)")
                failed += 1
                continue

            dataset.append({
                "question": question,
                "ground_truth": answer,
                "reference_contexts": [
                    f"[Source: {source}]\n{doc.page_content}"
                ],
                "source": source,
            })
            print(f" ✓")

        except json.JSONDecodeError:
            print(" ✗ (JSON parse error)")
            failed += 1
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(" ⏳ (rate limit — sleeping 10s)")
                time.sleep(10)
            else:
                print(f" ✗ ({e})")
            failed += 1

        # Small delay between calls to stay under rate limits
        time.sleep(0.5)

    print(f"\n  ✅ Generated {len(dataset)} rows  ({failed} failed/skipped)")

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved to {DATASET_FILE}\n")

    return dataset


# ── Step 2: Run RAG pipeline on each question ─────────────────────────

def run_pipeline_on_dataset(dataset: list[dict]) -> list[dict]:
    """
    Run each question through the LangGraph RAG pipeline.
    Collects (question, answer, contexts, ground_truth) for RAGAS scoring.
    """
    from graph_v2 import build_graph_v2

    print(f"\n{'='*60}")
    print(f"  Running RAG pipeline on {len(dataset)} questions")
    print(f"{'='*60}\n")

    graph = build_graph_v2()
    results = []
    failed = 0

    for i, row in enumerate(dataset):
        question = row["question"]
        print(f"  [{i+1}/{len(dataset)}] {question[:70]}...", end="", flush=True)

        try:
            state = graph.invoke({
                "question": question,
                "original_question": question,
                "chat_history": [],
                "documents": None,
                "answer": None,
                "route": None,
                "documents_relevant": None,
                "is_grounded": None,
                "is_useful": None,
                "generation_count": 0,
                "companies_compared": None,
                "grade_vector": None,
            })

            answer = state.get("answer", "")
            docs = state.get("documents") or []
            contexts = [
                f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
                for d in docs
            ]

            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts if contexts else ["No context retrieved"],
                "ground_truth": row["ground_truth"],
                "reference_contexts": row["reference_contexts"],
                "source": row.get("source", ""),
                "route": state.get("route", "unknown"),
                "is_grounded": state.get("is_grounded"),
            })
            print(f" ✓ ({state.get('route', '?')})")

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(" ⏳ (rate limit — sleeping 60s)")
                time.sleep(60)
                # retry once
                try:
                    state = graph.invoke({
                        "question": question,
                        "original_question": question,
                        "chat_history": [],
                        "documents": None,
                        "answer": None,
                        "route": None,
                        "documents_relevant": None,
                        "is_grounded": None,
                        "is_useful": None,
                        "generation_count": 0,
                        "companies_compared": None,
                        "grade_vector": None,
                    })
                    answer = state.get("answer", "")
                    docs = state.get("documents") or []
                    contexts = [
                        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
                        for d in docs
                    ]
                    results.append({
                        "question": question,
                        "answer": answer,
                        "contexts": contexts if contexts else ["No context retrieved"],
                        "ground_truth": row["ground_truth"],
                        "reference_contexts": row["reference_contexts"],
                        "source": row.get("source", ""),
                        "route": state.get("route", "unknown"),
                        "is_grounded": state.get("is_grounded"),
                    })
                    print(f" ✓ (retry ok)")
                except Exception:
                    print(f" ✗ (retry failed)")
                    failed += 1
            else:
                print(f" ✗ ({e})")
                failed += 1

        # Each pipeline run burns ~4-5k tokens (route+grade+generate+hallucination+usefulness).
        # Groq free tier cap is 12k tokens/min → max ~2-3 questions/min → need 20s+ gap.
        time.sleep(22.0)

    print(f"\n  ✅ Pipeline ran on {len(results)} questions  ({failed} failed)\n")

    # Save to disk so scoring can be retried without re-running the pipeline
    with open(PIPELINE_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  💾 Pipeline results saved to {PIPELINE_RESULTS_FILE}\n")

    return results


# ── Step 3: Score with RAGAS ──────────────────────────────────────────

def score_with_ragas(results: list[dict]) -> dict:
    """
    Run RAGAS metrics on the pipeline results.

    Metrics:
    - faithfulness: are all answer claims supported by retrieved contexts?
    - answer_relevancy: does the answer address the question?
    - context_recall: did retrieval get back the info needed to answer?
    - context_precision: of what was retrieved, how much was actually needed?
    """
    import math
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextRecall,
        ContextPrecision,
    )
    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
    from groq import AsyncGroq
    import instructor

    print(f"\n{'='*60}")
    print(f"  Scoring {len(results)} rows with RAGAS")
    print(f"{'='*60}\n")

    # Use async Groq client — RAGAS Collections metrics use agenerate() internally
    groq_client = instructor.from_groq(
        AsyncGroq(api_key=os.environ["GROQ_API_KEY"]),
        mode=instructor.Mode.JSON,
    )
    judge_llm = llm_factory(
        "openai/gpt-oss-120b",
        provider="groq",
        client=groq_client,
    )

    # Use the same embedding model as the pipeline (ragas>=0.4 native HF provider)
    embed_model = RagasHFEmbeddings(model="all-MiniLM-L6-v2", device="cpu")

    # Instantiate metrics (ragas>=0.4 Collections API — scored per-row, not via evaluate())
    faithfulness_metric = Faithfulness(llm=judge_llm)
    relevancy_metric = AnswerRelevancy(llm=judge_llm, embeddings=embed_model)
    recall_metric = ContextRecall(llm=judge_llm)
    precision_metric = ContextPrecision(llm=judge_llm)

    print("  Running RAGAS evaluation (this makes multiple LLM calls per row)...")
    print(f"  Expected time: ~3-6 min for {len(results)} rows\n")

    per_row = []
    all_scores = {"faithfulness": [], "answer_relevancy": [], "context_recall": [], "context_precision": []}

    for i, r in enumerate(results):
        q = r["question"]
        ans = r["answer"]
        ctxs = r["contexts"]
        gt = r["ground_truth"]
        print(f"  [{i+1}/{len(results)}] scoring: {q[:60]}...", end="", flush=True)

        row_scores = {}
        try:
            res = faithfulness_metric.score(user_input=q, response=ans, retrieved_contexts=ctxs)
            v = res.value
            row_scores["faithfulness"] = round(v, 4) if v is not None and not math.isnan(v) else None
        except Exception as e:
            print(f" [F-err:{e}]", end="")
            row_scores["faithfulness"] = None

        try:
            res = relevancy_metric.score(user_input=q, response=ans)
            v = res.value
            row_scores["answer_relevancy"] = round(v, 4) if v is not None and not math.isnan(v) else None
        except Exception as e:
            print(f" [AR-err:{e}]", end="")
            row_scores["answer_relevancy"] = None

        try:
            res = recall_metric.score(user_input=q, retrieved_contexts=ctxs, reference=gt)
            v = res.value
            row_scores["context_recall"] = round(v, 4) if v is not None and not math.isnan(v) else None
        except Exception as e:
            print(f" [CR-err:{e}]", end="")
            row_scores["context_recall"] = None

        try:
            res = precision_metric.score(user_input=q, reference=gt, retrieved_contexts=ctxs)
            v = res.value
            row_scores["context_precision"] = round(v, 4) if v is not None and not math.isnan(v) else None
        except Exception as e:
            print(f" [CP-err:{e}]", end="")
            row_scores["context_precision"] = None

        for k, v in row_scores.items():
            if v is not None:
                all_scores[k].append(v)

        print(f"  F={row_scores.get('faithfulness','?')}  AR={row_scores.get('answer_relevancy','?')}  CR={row_scores.get('context_recall','?')}  CP={row_scores.get('context_precision','?')}")

        per_row.append({
            "question": q,
            "source": r.get("source", ""),
            "route": r.get("route", ""),
            "is_grounded": r.get("is_grounded"),
            "scores": row_scores,
        })

        # Small pause between rows to avoid rate limits
        time.sleep(2.0)

    # Aggregate means
    agg = {k: round(sum(v) / len(v), 4) if v else None for k, v in all_scores.items()}

    return {"aggregate": agg, "per_row": per_row}


# ── Step 4: Save & print report ───────────────────────────────────────

def save_and_print_report(scores: dict, results: list[dict]):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_file = os.path.join(EVAL_DIR, f"scores_{timestamp}.json")

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(results),
        "aggregate_scores": scores["aggregate"],
        "per_row": scores["per_row"],
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    agg = scores["aggregate"]

    def fmt(v):
        return f"{v:<6}" if v is not None else "n/a   "

    print(f"\n{'='*60}")
    print(f"  RAGAS Evaluation Results")
    print(f"{'='*60}")
    print(f"  Questions evaluated : {len(results)}")
    print(f"  Run date            : {timestamp}")
    print()
    print(f"  Faithfulness        : {fmt(agg.get('faithfulness'))}  (are claims grounded in sources?)")
    print(f"  Answer Relevancy    : {fmt(agg.get('answer_relevancy'))}  (does answer address the question?)")
    print(f"  Context Recall      : {fmt(agg.get('context_recall'))}  (did retrieval get the right chunks?)")
    print(f"  Context Precision   : {fmt(agg.get('context_precision'))}  (were retrieved chunks all needed?)")
    print()

    # Surface worst-performing rows
    worst = sorted(
        [r for r in scores["per_row"] if r["scores"]],
        key=lambda r: sum(v for v in r["scores"].values() if v is not None)
    )[:5]
    if worst:
        print(f"  5 lowest-scoring questions:")
        for r in worst:
            s = r["scores"]
            print(f"    [{r['route']:8}] F={s.get('faithfulness','?'):.2f}  "
                  f"AR={s.get('answer_relevancy','?'):.2f}  "
                  f"CR={s.get('context_recall','?'):.2f}  "
                  f"CP={s.get('context_precision','?'):.2f}")
            print(f"              {r['question'][:80]}")

    print(f"\n  💾 Full report saved to {out_file}")
    print(f"{'='*60}\n")

    return out_file


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EarningsLens RAGAS evaluation pipeline."
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate synthetic eval dataset from Pinecone chunks.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Run RAG pipeline + RAGAS scoring on the eval dataset.",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of Q&A pairs to generate (default: 20). Groq free tier supports ~2-3 pipeline runs/min.",
    )
    parser.add_argument(
        "--pipeline-only", action="store_true",
        help=f"Run only the RAG pipeline (no RAGAS scoring). Saves results to {PIPELINE_RESULTS_FILE}. "
             "Use --score-only the next day to score without re-running the pipeline.",
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help=f"Skip the pipeline, load saved results from {PIPELINE_RESULTS_FILE}, and re-run RAGAS scoring only.",
    )
    args = parser.parse_args()

    if not args.generate and not args.run and not args.score_only and not args.pipeline_only:
        parser.print_help()
        return

    dataset = None

    if args.generate:
        dataset = generate_dataset(n=args.n)

    if args.pipeline_only:
        if dataset is None:
            if not os.path.exists(DATASET_FILE):
                print(f"❌ No dataset found at {DATASET_FILE}. Run with --generate first.")
                return
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"  Loaded {len(dataset)} rows from {DATASET_FILE}")
        run_pipeline_on_dataset(dataset)
        print(f"  Pipeline complete. Run --score-only tomorrow to score with a fresh token budget.")
        return

    if args.score_only:
        if not os.path.exists(PIPELINE_RESULTS_FILE):
            print(f"❌ No pipeline results found at {PIPELINE_RESULTS_FILE}. Run with --pipeline-only first.")
            return
        with open(PIPELINE_RESULTS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} pipeline results from {PIPELINE_RESULTS_FILE}")
        scores = score_with_ragas(results)
        save_and_print_report(scores, results)
        return

    if args.run:
        if dataset is None:
            if not os.path.exists(DATASET_FILE):
                print(f"❌ No dataset found at {DATASET_FILE}. Run with --generate first.")
                return
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"  Loaded {len(dataset)} rows from {DATASET_FILE}")

        results = run_pipeline_on_dataset(dataset)
        scores = score_with_ragas(results)
        save_and_print_report(scores, results)


if __name__ == "__main__":
    main()
