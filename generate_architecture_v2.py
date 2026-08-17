"""
generate_architecture_v2.py — Full system architecture map for EarningsLens v2 + GraphRAG
Saves output as architecture_map_v2.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Canvas ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(26, 30))
ax.set_xlim(0, 26)
ax.set_ylim(0, 30)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# ── Colour palette ──────────────────────────────────────────────────────────
C = {
    "bg":              "#0d1117",
    "panel_bg":        "#0f1923",
    "panel_border":    "#1e3a5f",
    "text":            "#f0f0f0",
    "subtext":         "#94a3b8",
    "arrow":           "#64748b",

    # Node fills + borders
    "start_end_fill":  "#1a1a2e",
    "start_end_edge":  "#e94560",

    "router_fill":     "#16213e",
    "router_edge":     "#f5a623",

    "retrieval_fill":  "#0f3460",
    "retrieval_edge":  "#4fc3f7",

    "llm_fill":        "#1b1b3a",
    "llm_edge":        "#a78bfa",

    "quality_fill":    "#0f2a1a",
    "quality_edge":    "#4ade80",

    "rewrite_fill":    "#2a1500",
    "rewrite_edge":    "#fb923c",

    # NEW — GraphRAG nodes
    "graph_node_fill": "#1a1500",
    "graph_node_edge": "#fbbf24",   # amber / gold

    "graph_store_fill":"#071a1a",
    "graph_store_edge":"#2dd4bf",   # teal

    "offline_fill":    "#150a20",
    "offline_edge":    "#c084fc",   # purple

    # Edge colours
    "yes":             "#4ade80",
    "no":              "#f87171",
    "graph_flow":      "#fbbf24",
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def panel(ax, x, y, w, h, title, border_color, title_color=None):
    """Draw a labelled section panel."""
    bg = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.3",
        linewidth=1.8, edgecolor=border_color,
        facecolor=C["panel_bg"], zorder=1, alpha=0.85)
    ax.add_patch(bg)
    ax.text(x + w/2, y + h - 0.38, title,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=title_color or border_color, zorder=4,
            fontfamily="monospace")

def node(ax, x, y, w, h, label, sublabel, fill, edge, radius=0.3):
    """Draw a rounded rectangle node."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=2.2, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.13, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=C["text"],
                zorder=4, fontfamily="monospace")
        ax.text(x, y - 0.25, sublabel, ha="center", va="center",
                fontsize=7.2, color=C["subtext"], zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=C["text"],
                zorder=4, fontfamily="monospace")

def storage(ax, x, y, w, h, label, sublabel, fill, edge):
    """Draw a cylinder-style storage node (rounded rect with top arc hint)."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.25",
        linewidth=2.0, edgecolor=edge, facecolor=fill,
        linestyle="dashed", zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.13, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=edge, zorder=4, fontfamily="monospace")
        ax.text(x, y - 0.25, sublabel, ha="center", va="center",
                fontsize=7.2, color=C["subtext"], zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=edge, zorder=4, fontfamily="monospace")

def pill(ax, x, y, w, h, label, fill, edge, fontsize=8):
    """Draw a small pill/tag node."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        linewidth=1.5, edgecolor=edge, facecolor=fill, zorder=3, alpha=0.9)
    ax.add_patch(ax.add_patch(box) or box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=C["text"], zorder=4)

def arr(ax, x1, y1, x2, y2, color="#64748b", lw=1.8, label="", lside="right", rad=0):
    """Draw a straight or curved arrow."""
    style = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=15, connectionstyle=style), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ox = 0.28 if lside == "right" else -0.28
        ax.text(mx + ox, my, label, ha="left" if lside == "right" else "right",
                va="center", fontsize=7.5, color=color, fontweight="bold", zorder=6)

def circle_node(ax, x, y, r, label, fill, edge, fontsize=8.5):
    c = plt.Circle((x, y), r, color=fill, zorder=3, linewidth=2.2,
                   linestyle="solid")
    c.set_edgecolor(edge)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="#0d1117", zorder=4, fontfamily="monospace")

# ═══════════════════════════════════════════════════════════════════════════════
#  TITLE
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(13, 29.4, "EarningsLens — Adaptive RAG Architecture  (v2 + GraphRAG)",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=C["text"], zorder=6)
ax.text(13, 28.9,
        "LangGraph  •  Pinecone  •  Groq LLaMA-3.3-70b  •  NetworkX Knowledge Graph",
        ha="center", va="center", fontsize=9.5, color=C["subtext"], zorder=6)

# ═══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL — FRONTEND
# ═══════════════════════════════════════════════════════════════════════════════
panel(ax, 0.2, 19.5, 4.8, 9.0, "FRONTEND  (app_v2.py)", "#e94560")

node(ax, 2.6, 27.8, 3.8, 0.75, "Streamlit Chat Interface", None,
     "#1e0a0a", "#e94560")

# UI sub-components
for i, (lbl, sub) in enumerate([
    ("PDF Upload",       "real-time ingest → Pinecone"),
    ("Source Citations", "per-message doc references"),
    ("Reasoning Path",   "graph_path expander"),
]):
    node(ax, 2.6, 26.7 - i*1.0, 3.6, 0.7, lbl, sub, "#160808", "#e94560", radius=0.2)

ax.text(2.6, 23.5, "Sidebar Metrics", ha="center", fontsize=8.5,
        fontweight="bold", color="#f5a623", zorder=4)
for i, lbl in enumerate(["Total queries", "Avg latency (ms)",
                          "Grounded %", "MRR"]):
    ax.text(2.6, 23.1 - i*0.42, f"• {lbl}", ha="center", fontsize=7.5,
            color=C["subtext"], zorder=4)

node(ax, 2.6, 20.5, 3.6, 0.7, "Company Clarifier", "pre-query ambiguity check",
     "#160808", "#e94560", radius=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL — DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════
panel(ax, 0.2, 6.5, 4.8, 12.5, "DATA LAYER", "#2dd4bf")

node(ax, 2.6, 18.2, 3.8, 0.75, "data/  (local corpus)", None,
     "#051a1a", "#2dd4bf")
ax.text(2.6, 17.55, "• 20 .txt  earnings transcripts", ha="center",
        fontsize=7.5, color=C["subtext"], zorder=4)
ax.text(2.6, 17.2, "• 2 .pdf  SEC filings (10-Q)", ha="center",
        fontsize=7.5, color=C["subtext"], zorder=4)
ax.text(2.6, 16.85, "• AAPL  MSFT  AMZN  NVDA  TSLA  +", ha="center",
        fontsize=7.5, color=C["subtext"], zorder=4)

node(ax, 2.6, 15.8, 3.8, 0.75, "ingest_v2.py", "chunk → embed → upsert",
     "#051a1a", "#2dd4bf")
ax.text(2.6, 15.1, "RecursiveCharacterTextSplitter", ha="center",
        fontsize=7.2, color=C["subtext"], style="italic", zorder=4)
ax.text(2.6, 14.72, "size=1000  overlap=200", ha="center",
        fontsize=7.2, color=C["subtext"], style="italic", zorder=4)

node(ax, 2.6, 13.7, 3.8, 0.75, "ingest_hf.py", "HuggingFace → Pinecone",
     "#051a1a", "#2dd4bf")
ax.text(2.6, 13.0, "Bose345/sp500_earnings_transcripts", ha="center",
        fontsize=7.2, color=C["subtext"], style="italic", zorder=4)
ax.text(2.6, 12.65, "252 tickers  •  1,326 documents", ha="center",
        fontsize=7.2, color=C["subtext"], style="italic", zorder=4)

storage(ax, 2.6, 11.5, 3.6, 0.7, "ingested_manifest.json",
        "chunk counts + ticker metadata",
        "#051a1a", "#2dd4bf")

node(ax, 2.6, 10.2, 3.8, 0.75, "eval_ragas.py", "RAGAS offline evaluation",
     "#051a1a", "#2dd4bf")
for i, lbl in enumerate(["Faithfulness", "Answer Relevance",
                          "Context Precision", "Context Recall"]):
    ax.text(2.6, 9.6 - i*0.38, f"• {lbl}", ha="center", fontsize=7.2,
            color=C["subtext"], zorder=4)

storage(ax, 2.6, 7.8, 3.6, 0.7, "eval/scores_*.json",
        "RAGAS score snapshots", "#051a1a", "#2dd4bf")

# ═══════════════════════════════════════════════════════════════════════════════
#  CENTRE PANEL — LANGGRAPH PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
panel(ax, 5.3, 4.5, 13.2, 24.0,
      "LANGGRAPH ADAPTIVE RAG PIPELINE  (graph_v2.py  •  nodes_v2.py)", "#4fc3f7")

# Node positions (x=11.9 centre spine)
CX = 11.9
NODES = {
    "START":               (CX, 27.6),
    "route_question":      (CX, 25.9),
    "compare":             (7.2, 23.0),
    "retrieve":            (CX, 23.0),
    "direct_answer":       (16.5, 23.0),
    "retrieve_graph":      (CX, 21.0),   # NEW
    "grade_documents":     (CX, 19.0),
    "generate":            (CX, 16.7),
    "check_hallucination": (CX, 14.4),
    "check_usefulness":    (CX, 12.1),
    "rewrite_query":       (17.0, 15.4),
    "END":                 (CX, 9.8),
    "END_giveup":          (7.2, 16.7),
}

NW, NH = 3.6, 0.85

# ── Draw pipeline nodes ──────────────────────────────────────────────────────

circle_node(ax, *NODES["START"], 0.52, "START", C["start_end_fill"], C["start_end_edge"], 8)

node(ax, *NODES["route_question"], NW+0.6, NH,
     "route_question", "Node 1 — Router LLM",
     C["router_fill"], C["router_edge"])

node(ax, *NODES["compare"], NW+0.6, NH+0.35,
     "compare_companies",
     "Node 9 — Extract · retrieve\nper-company · generate",
     C["llm_fill"], C["llm_edge"])

node(ax, *NODES["retrieve"], NW, NH,
     "retrieve", "Node 2 — Pinecone  top-k = 5",
     C["retrieval_fill"], C["retrieval_edge"])

node(ax, *NODES["direct_answer"], NW+0.2, NH,
     "direct_answer", "Node 5 — General knowledge LLM",
     C["llm_fill"], C["llm_edge"])

# ── NEW: retrieve_graph ──────────────────────────────────────────────────────
node(ax, *NODES["retrieve_graph"], NW+0.6, NH+0.15,
     "retrieve_graph  ✦ NEW",
     "Node 2b — Graph traversal · format context",
     C["graph_node_fill"], C["graph_node_edge"])

# ── Remaining pipeline nodes ─────────────────────────────────────────────────
node(ax, *NODES["grade_documents"], NW+0.4, NH,
     "grade_documents", "Node 3 — Relevance grader LLM",
     C["router_fill"], C["router_edge"])

node(ax, *NODES["generate"], NW, NH,
     "generate", "Node 4 — Answer generation LLM  ✦ enriched",
     C["llm_fill"], C["llm_edge"])

node(ax, *NODES["check_hallucination"], NW+0.6, NH,
     "check_hallucination", "Node 6 — Fact-checker LLM",
     C["quality_fill"], C["quality_edge"])

node(ax, *NODES["check_usefulness"], NW+0.6, NH,
     "check_usefulness", "Node 7 — Quality evaluator LLM",
     C["quality_fill"], C["quality_edge"])

node(ax, *NODES["rewrite_query"], NW+0.2, NH,
     "rewrite_query", "Node 8 — Query rewriter LLM",
     C["rewrite_fill"], C["rewrite_edge"])

circle_node(ax, *NODES["END"], 0.52, "END", C["start_end_fill"], C["start_end_edge"], 8)

cx3, cy3 = NODES["END_giveup"]
end2 = plt.Circle((cx3, cy3), 0.52, color="#3b0a0a", zorder=3,
                   linewidth=2, linestyle="solid")
end2.set_edgecolor("#dc2626")
ax.add_patch(end2)
ax.text(cx3, cy3, "END\n(give up)", ha="center", va="center",
        fontsize=7.2, fontweight="bold", color="#fca5a5", zorder=4, fontfamily="monospace")

# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE EDGES
# ═══════════════════════════════════════════════════════════════════════════════

# START → route_question
arr(ax, CX, 27.08, CX, 26.33, color=C["arrow"])

# route_question → compare
arr(ax, CX-0.5, 25.52, NODES["compare"][0]+0.5, NODES["compare"][1]+0.45,
    color=C["llm_edge"], label='"compare"', lside="left", rad=-0.15)

# route_question → retrieve
arr(ax, CX, 25.48, CX, 23.43, color=C["retrieval_edge"], label='"retrieve"')

# route_question → direct_answer
arr(ax, CX+0.5, 25.52, NODES["direct_answer"][0]-0.45, NODES["direct_answer"][1]+0.4,
    color=C["llm_edge"], label='"direct"', rad=0.15)

# compare → END
ax.annotate("", xy=(NODES["END"][0]-0.3, NODES["END"][1]+0.52),
            xytext=(NODES["compare"][0], NODES["compare"][1]-0.6),
            arrowprops=dict(arrowstyle="-|>", color=C["yes"], lw=1.8,
                            mutation_scale=14, connectionstyle="arc3,rad=0.3"), zorder=5)
ax.text(6.0, 13.2, "answer ready", fontsize=7, color=C["yes"], fontweight="bold", zorder=6)

# direct_answer → END
ax.annotate("", xy=(NODES["END"][0]+0.35, NODES["END"][1]+0.52),
            xytext=(NODES["direct_answer"][0], NODES["direct_answer"][1]-0.43),
            arrowprops=dict(arrowstyle="-|>", color=C["yes"], lw=1.8,
                            mutation_scale=14, connectionstyle="arc3,rad=-0.25"), zorder=5)

# retrieve → retrieve_graph  (NEW)
arr(ax, CX, 22.58, CX, 21.48, color=C["graph_node_edge"], lw=2.2)

# retrieve_graph → grade_documents  (NEW)
arr(ax, CX, 20.52, CX, 19.43, color=C["graph_node_edge"], lw=2.2)

# grade_documents → generate
arr(ax, CX, 18.58, CX, 17.13, color=C["yes"], label="relevant ✓")

# grade_documents → rewrite
arr(ax, NODES["grade_documents"][0]+2.0, NODES["grade_documents"][1],
    NODES["rewrite_query"][0]-0.3, NODES["rewrite_query"][1]+0.35,
    color=C["no"], rad=0.2, label="not relevant\n(gen < 2)")

# grade_documents → END_giveup
arr(ax, NODES["grade_documents"][0]-1.95, NODES["grade_documents"][1],
    NODES["END_giveup"][0]+0.5, NODES["END_giveup"][1]+0.1,
    color="#f87171", label="not relevant\n(gen ≥ 2)", lside="left")

# generate → check_hallucination
arr(ax, CX, 16.28, CX, 14.83, color=C["llm_edge"])

# check_hallucination → check_usefulness
arr(ax, CX, 13.98, CX, 12.53, color=C["yes"], label="grounded ✓")

# check_hallucination → rewrite
arr(ax, NODES["check_hallucination"][0]+2.0, NODES["check_hallucination"][1],
    NODES["rewrite_query"][0]-0.3, NODES["rewrite_query"][1]-0.05,
    color=C["no"], rad=0.05, label="not grounded\n(gen < 2)")

# check_hallucination → END (gen≥2)
ax.annotate("", xy=(NODES["END"][0]-0.5, NODES["END"][1]+0.52),
            xytext=(NODES["check_hallucination"][0]-1.8, NODES["check_hallucination"][1]-0.3),
            arrowprops=dict(arrowstyle="-|>", color="#f87171", lw=1.6,
                            mutation_scale=13, connectionstyle="arc3,rad=-0.2"), zorder=5)
ax.text(8.5, 11.6, "not grounded\n(gen ≥ 2)", ha="right", fontsize=7,
        color="#f87171", fontweight="bold", zorder=6)

# check_usefulness → END
arr(ax, CX, 11.63, CX, 10.32, color=C["yes"], label="useful ✓")

# check_usefulness → rewrite
arr(ax, NODES["check_usefulness"][0]+2.0, NODES["check_usefulness"][1],
    NODES["rewrite_query"][0]-0.3, NODES["rewrite_query"][1]-0.45,
    color=C["no"], rad=-0.2, label="not useful\n(gen < 2)")

# check_usefulness → END (gen≥2)
ax.annotate("", xy=(NODES["END"][0]+0.48, NODES["END"][1]+0.48),
            xytext=(NODES["check_usefulness"][0]+0.6, NODES["check_usefulness"][1]-0.42),
            arrowprops=dict(arrowstyle="-|>", color="#f87171", lw=1.6,
                            mutation_scale=13, connectionstyle="arc3,rad=-0.15"), zorder=5)

# rewrite_query → retrieve  (loop back)
ax.annotate("",
    xy=(NODES["retrieve"][0]+1.85, NODES["retrieve"][1]),
    xytext=(NODES["rewrite_query"][0], NODES["rewrite_query"][1]+0.43),
    arrowprops=dict(arrowstyle="-|>", color=C["rewrite_edge"], lw=2.2,
                    mutation_scale=17, connectionstyle="arc3,rad=-0.5"), zorder=5)
ax.text(18.3, 18.8, "retry\n(loop)", ha="center", va="center",
        fontsize=8, color=C["rewrite_edge"], fontweight="bold", zorder=6)

# ═══════════════════════════════════════════════════════════════════════════════
#  RIGHT TOP — EXTERNAL SERVICES
# ═══════════════════════════════════════════════════════════════════════════════
panel(ax, 19.3, 22.5, 6.3, 7.0, "EXTERNAL SERVICES", "#a78bfa")

node(ax, 22.45, 28.2, 5.0, 0.7, "Pinecone (Cloud)", "earningslens-v2  •  384-dim cosine",
     "#180b2e", "#a78bfa")
ax.text(22.45, 27.6, "AWS us-east-1 serverless", ha="center",
        fontsize=7.2, color=C["subtext"], style="italic", zorder=4)

node(ax, 22.45, 26.6, 5.0, 0.7, "Groq API", "openai/gpt-oss-120b  •  T=0",
     "#180b2e", "#a78bfa")
ax.text(22.45, 26.0, "Router · Grader · Generate · Hallucination\nUsefulness · Rewrite · Compare · Extract",
        ha="center", fontsize=7, color=C["subtext"], zorder=4)

node(ax, 22.45, 24.7, 5.0, 0.7, "HuggingFace Embeddings",
     "all-MiniLM-L6-v2  •  384 dims  •  CPU",
     "#180b2e", "#a78bfa")

node(ax, 22.45, 23.4, 5.0, 0.7, "LangSmith (optional)",
     "LANGCHAIN_TRACING_V2=true",
     "#180b2e", "#a78bfa")

# ═══════════════════════════════════════════════════════════════════════════════
#  RIGHT MIDDLE — KNOWLEDGE GRAPH (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
panel(ax, 19.3, 12.8, 6.3, 9.3, "KNOWLEDGE GRAPH  ✦ NEW", C["graph_node_edge"])

node(ax, 22.45, 21.3, 5.2, 0.75, "build_graph.py  (offline)",
     "run once after ingestion",
     C["offline_fill"], C["offline_edge"])

# build steps
build_steps = [
    "1. Load data/*.txt + data/*.pdf",
    "2. Sample 5 chunks / file",
    "3. LLM entity extraction (Groq)",
    "4. Build NetworkX DiGraph",
    "5. Save → graph.json",
]
for i, s in enumerate(build_steps):
    ax.text(22.45, 20.6 - i*0.42, s, ha="center", fontsize=7,
            color=C["subtext"], zorder=4)

ax.text(22.45, 18.6, "Extracts:", ha="center", fontsize=7.8,
        fontweight="bold", color=C["graph_node_edge"], zorder=4)
entities = ["Executives · roles", "Financial metrics · values",
            "Products · services", "Macro themes (vocab=15)",
            "Competitor mentions"]
for i, e in enumerate(entities):
    ax.text(22.45, 18.2 - i*0.38, f"• {e}", ha="center", fontsize=7,
            color=C["subtext"], zorder=4)

storage(ax, 22.45, 15.8, 5.0, 0.75, "graph.json",
        "NetworkX DiGraph  •  node_link format",
        C["graph_store_fill"], C["graph_store_edge"])

ax.text(22.45, 15.1, "Nodes: company · document · executive", ha="center",
        fontsize=7, color=C["subtext"], zorder=4)
ax.text(22.45, 14.72, "        metric · product · theme", ha="center",
        fontsize=7, color=C["subtext"], zorder=4)
ax.text(22.45, 14.34, "Edges: filed · speaker · reports · discusses", ha="center",
        fontsize=7, color=C["subtext"], zorder=4)
ax.text(22.45, 13.96, "        competes_with · next_quarter · theme_of", ha="center",
        fontsize=7, color=C["subtext"], zorder=4)

# Arrow: build_graph.py → graph.json
arr(ax, 22.45, 20.92, 22.45, 16.18,
    color=C["offline_edge"], lw=1.6)

# Arrow: graph.json → retrieve_graph (horizontal into main pipeline)
ax.annotate("",
    xy=(NODES["retrieve_graph"][0]+2.0, NODES["retrieve_graph"][1]),
    xytext=(22.45 - 2.55, 15.8),
    arrowprops=dict(arrowstyle="-|>", color=C["graph_node_edge"], lw=2.4,
                    mutation_scale=16,
                    connectionstyle="arc3,rad=-0.25"), zorder=5)
ax.text(20.3, 17.6, "graph_context\n→ state", ha="center", va="center",
        fontsize=7.5, color=C["graph_node_edge"], fontweight="bold", zorder=6)

# Arrow: data/ → build_graph.py (from data layer to knowledge graph panel)
ax.annotate("",
    xy=(19.3, 21.3),
    xytext=(5.1, 18.2),
    arrowprops=dict(arrowstyle="-|>", color=C["offline_edge"], lw=1.6,
                    mutation_scale=13, connectionstyle="arc3,rad=-0.15"), zorder=2)
ax.text(12.2, 20.2, "raw text", ha="center", fontsize=7,
        color=C["offline_edge"], style="italic", zorder=6)

# ═══════════════════════════════════════════════════════════════════════════════
#  RIGHT BOTTOM — OBSERVABILITY
# ═══════════════════════════════════════════════════════════════════════════════
panel(ax, 19.3, 5.5, 6.3, 6.9, "OBSERVABILITY", "#4ade80")

storage(ax, 22.45, 11.5, 5.0, 0.7, "logs/query_log.jsonl",
        "per-query append log",
        "#071a0e", "#4ade80")
log_fields = ["timestamp  •  question", "latency_ms  •  route",
              "docs_retrieved  •  is_grounded", "grade_vector  •  graph_context_used"]
for i, f in enumerate(log_fields):
    ax.text(22.45, 10.9 - i*0.4, f"  {f}", ha="center", fontsize=7,
            color=C["subtext"], zorder=4)

node(ax, 22.45, 9.5, 5.0, 0.7, "RAG Performance Page",
     "pages/RAG_Performance.py",
     "#071a0e", "#4ade80")
metrics_list = ["Total queries", "Avg latency (ms)",
                "Grounded %", "MRR (Mean Reciprocal Rank)"]
for i, m in enumerate(metrics_list):
    ax.text(22.45, 8.95 - i*0.38, f"• {m}", ha="center", fontsize=7,
            color=C["subtext"], zorder=4)

storage(ax, 22.45, 7.0, 5.0, 0.7, "eval/scores_*.json",
        "RAGAS snapshot scores",
        "#071a0e", "#4ade80")
ax.text(22.45, 6.35, "Faithfulness  •  Answer Relevance", ha="center",
        fontsize=7, color=C["subtext"], zorder=4)
ax.text(22.45, 5.98, "Context Precision  •  Context Recall", ha="center",
        fontsize=7, color=C["subtext"], zorder=4)

# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH CONTEXT CALLOUT (annotation box near generate)
# ═══════════════════════════════════════════════════════════════════════════════
callout_x, callout_y = 6.2, 16.2
box = FancyBboxPatch((callout_x - 2.55, callout_y - 1.75), 5.1, 1.9,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    linewidth=1.5, edgecolor=C["graph_node_edge"],
    facecolor="#100d00", zorder=3, alpha=0.92)
ax.add_patch(box)
ax.text(callout_x, callout_y + 0.62, "graph_context injected into generate:",
        ha="center", va="center", fontsize=7, fontweight="bold",
        color=C["graph_node_edge"], zorder=4)
sample_lines = [
    "[Apple Inc. (AAPL)]",
    "  Metrics: revenue 94.9B (Q1 2025)",
    "  Speakers: Tim Cook (CEO)",
    "  Themes: AI adoption, margin expansion",
    "[Cross-company themes]",
    "  'AI adoption' also by: MSFT, NVDA",
]
for i, line in enumerate(sample_lines):
    color = C["graph_node_edge"] if line.startswith("[") else "#c0c0a0"
    ax.text(callout_x, callout_y + 0.2 - i * 0.33, line,
            ha="center", va="center", fontsize=6.5, color=color,
            fontfamily="monospace", zorder=4)

# Dotted arrow from callout to generate node
ax.annotate("",
    xy=(NODES["generate"][0] - 1.85, NODES["generate"][1] + 0.1),
    xytext=(callout_x + 2.55, callout_y - 0.4),
    arrowprops=dict(arrowstyle="-|>", color=C["graph_node_edge"], lw=1.4,
                    mutation_scale=11, linestyle="dashed",
                    connectionstyle="arc3,rad=0.2"), zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOOP GUARD ANNOTATION
# ═══════════════════════════════════════════════════════════════════════════════
loop_box = FancyBboxPatch((5.6, 5.0), 5.4, 1.2,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    linewidth=1.5, edgecolor="#4b5563", facecolor="#1f2937", zorder=3)
ax.add_patch(loop_box)
ax.text(8.3, 5.9, "Loop guard: generation_count",
        ha="center", va="center", fontsize=8.5, fontweight="bold",
        color="#f59e0b", zorder=4)
ax.text(8.3, 5.45, "Increments each generate() call.",
        ha="center", va="center", fontsize=7.5, color=C["subtext"], zorder=4)
ax.text(8.3, 5.08, "If ≥ 2 → all retry paths go to END instead.",
        ha="center", va="center", fontsize=7.5, color=C["subtext"], zorder=4)

# ═══════════════════════════════════════════════════════════════════════════════
#  LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
lx, ly = 11.4, 4.8
ax.text(lx, ly, "LEGEND", fontsize=8.5, fontweight="bold",
        color=C["subtext"], zorder=6)
legend_items = [
    (C["router_edge"],      "Router / Grader LLM node"),
    (C["retrieval_edge"],   "Vector retrieval (Pinecone)"),
    (C["graph_node_edge"],  "GraphRAG node / flow  ✦ NEW"),
    (C["graph_store_edge"], "Knowledge graph store  ✦ NEW"),
    (C["llm_edge"],         "LLM generation node"),
    (C["quality_edge"],     "Quality-check node"),
    (C["rewrite_edge"],     "Query rewrite / loop"),
    (C["offline_edge"],     "Offline build pipeline  ✦ NEW"),
    (C["yes"],              "Success branch"),
    (C["no"],               "Failure / retry branch (gen < 2)"),
    ("#f87171",             "Hard stop (gen ≥ 2 limit)"),
]
for i, (color, label) in enumerate(legend_items):
    y = ly - 0.52 - i * 0.5
    ax.add_patch(plt.Rectangle((lx, y - 0.12), 0.32, 0.28, color=color, zorder=6))
    ax.text(lx + 0.52, y + 0.02, label, fontsize=7.5,
            color="#d0d0d0", va="center", zorder=6)

# ── Version badge ────────────────────────────────────────────────────────────
badge = FancyBboxPatch((0.2, 0.15), 3.2, 0.65,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    linewidth=1.5, edgecolor=C["graph_node_edge"],
    facecolor="#1a1500", zorder=6)
ax.add_patch(badge)
ax.text(1.8, 0.48, "✦ GraphRAG added in v2", ha="center", va="center",
        fontsize=8, fontweight="bold", color=C["graph_node_edge"], zorder=7)

ax.text(25.8, 0.25, "EarningsLens  •  architecture_map_v2",
        ha="right", va="center", fontsize=7.5, color="#4b5563", zorder=6)

# ── Save ──────────────────────────────────────────────────────────────────────
out = r"C:\Users\PC\Desktop\earningslens\architecture_map_v2.png"
plt.tight_layout(pad=0)
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print(f"Saved: {out}")
