"""
generate_flowchart.py — Visual flowchart of the EarningsLens v2 LangGraph
Saves output as earningslens_graph_v2.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Canvas ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 22))
ax.set_xlim(0, 18)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# ── Colour palette ────────────────────────────────────────────────────
C = {
    "start_end":   "#1a1a2e",
    "start_edge":  "#e94560",
    "router":      "#16213e",
    "router_edge": "#f5a623",
    "retrieval":   "#0f3460",
    "retrieval_edge": "#4fc3f7",
    "llm":         "#1b1b3a",
    "llm_edge":    "#a78bfa",
    "quality":     "#1a2e1a",
    "quality_edge":"#4ade80",
    "rewrite":     "#2e1a0f",
    "rewrite_edge":"#fb923c",
    "text":        "#f0f0f0",
    "arrow":       "#94a3b8",
    "yes":         "#4ade80",
    "no":          "#f87171",
    "bg":          "#0f1117",
}

# ── Node helper ───────────────────────────────────────────────────────
def node(ax, x, y, w, h, label, sublabel, fill, edge, text_color="#f0f0f0", radius=0.35):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=2.2, edgecolor=edge, facecolor=fill, zorder=3
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.12, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=text_color, zorder=4,
                fontfamily="monospace")
        ax.text(x, y - 0.28, sublabel, ha="center", va="center",
                fontsize=7.5, color="#94a3b8", zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=text_color, zorder=4,
                fontfamily="monospace")

def diamond(ax, x, y, w, h, label, fill, edge):
    """Draw a diamond shape for decision nodes."""
    dx, dy = w/2, h/2
    pts = [(x, y+dy), (x+dx, y), (x, y-dy), (x-dx, y)]
    from matplotlib.patches import Polygon
    diamond_patch = Polygon(pts, closed=True, linewidth=2.2,
                            edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(diamond_patch)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#f0f0f0", zorder=4,
            fontfamily="monospace")

def arrow(ax, x1, y1, x2, y2, color="#94a3b8", lw=1.8, label="", label_side="right"):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=16, connectionstyle="arc3,rad=0"),
        zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        offset_x = 0.25 if label_side == "right" else -0.25
        ax.text(mx + offset_x, my, label, ha="left" if label_side == "right" else "right",
                va="center", fontsize=8, color=color, fontweight="bold", zorder=6)

def curved_arrow(ax, x1, y1, x2, y2, color="#94a3b8", lw=1.8, label="",
                 rad=0.3, label_side="right"):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=16, connectionstyle=f"arc3,rad={rad}"),
        zorder=5)
    if label:
        mx = (x1 + x2) / 2 + (0.4 if label_side == "right" else -0.4)
        my = (y1 + y2) / 2
        ax.text(mx, my, label, ha="left" if label_side == "right" else "right",
                va="center", fontsize=8, color=color, fontweight="bold", zorder=6)

# ═══════════════════════════════════════════════════════════════════════
#  NODE POSITIONS
# ═══════════════════════════════════════════════════════════════════════
#  x-axis: 0..18,  y-axis: 0..22 (top = high y)

POS = {
    "START":               (9,  21),
    "route_question":      (9,  19.2),
    "compare_companies":   (3,  16.5),
    "retrieve":            (9,  16.5),
    "direct_answer":       (15, 16.5),
    "grade_documents":     (9,  14),
    "generate":            (9,  11.5),
    "check_hallucination": (9,  9),
    "check_usefulness":    (9,  6.5),
    "rewrite_query":       (14.5, 11),
    "END":                 (9,  4),
    "END_giveup":          (3,  11.5),
}

NW = 3.6   # node width
NH = 0.9   # node height
DW = 3.2   # decision diamond half-width
DH = 0.7   # decision diamond half-height

# ═══════════════════════════════════════════════════════════════════════
#  DRAW NODES
# ═══════════════════════════════════════════════════════════════════════

# START
cx, cy = POS["START"]
start_patch = plt.Circle((cx, cy), 0.55, color=C["start_edge"], zorder=3)
ax.add_patch(start_patch)
ax.text(cx, cy, "START", ha="center", va="center", fontsize=9,
        fontweight="bold", color="#0f1117", zorder=4, fontfamily="monospace")

# route_question
node(ax, *POS["route_question"], NW+0.4, NH, "route_question",
     "Node 1 — Router LLM", C["router"], C["router_edge"])

# compare_companies
node(ax, *POS["compare_companies"], NW+0.5, NH+0.3, "compare_companies",
     "Node 9 — Extract companies,\nretrieve per-company, generate", C["llm"], C["llm_edge"])

# retrieve
node(ax, *POS["retrieve"], NW, NH, "retrieve",
     "Node 2 — Pinecone top-k=5", C["retrieval"], C["retrieval_edge"])

# direct_answer
node(ax, *POS["direct_answer"], NW, NH, "direct_answer",
     "Node 5 — General knowledge LLM", C["llm"], C["llm_edge"])

# grade_documents
node(ax, *POS["grade_documents"], NW+0.4, NH, "grade_documents",
     "Node 3 — Relevance grader LLM", C["router"], C["router_edge"])

# generate
node(ax, *POS["generate"], NW, NH, "generate",
     "Node 4 — Answer generation LLM", C["llm"], C["llm_edge"])

# check_hallucination
node(ax, *POS["check_hallucination"], NW+0.5, NH, "check_hallucination",
     "Node 6 — Fact-checker LLM", C["quality"], C["quality_edge"])

# check_usefulness
node(ax, *POS["check_usefulness"], NW+0.5, NH, "check_usefulness",
     "Node 7 — Quality evaluator LLM", C["quality"], C["quality_edge"])

# rewrite_query
node(ax, *POS["rewrite_query"], NW+0.2, NH, "rewrite_query",
     "Node 8 — Query rewriter LLM", C["rewrite"], C["rewrite_edge"])

# END (main)
cx2, cy2 = POS["END"]
end_patch = plt.Circle((cx2, cy2), 0.55, color=C["start_edge"], zorder=3)
ax.add_patch(end_patch)
ax.text(cx2, cy2, "END", ha="center", va="center", fontsize=9,
        fontweight="bold", color="#0f1117", zorder=4, fontfamily="monospace")

# END give_up (from grade_documents when gen_count >= 2)
cx3, cy3 = POS["END_giveup"]
end2_patch = plt.Circle((cx3, cy3), 0.55, color="#6b1a1a", zorder=3)
ax.add_patch(end2_patch)
ax.text(cx3, cy3, "END\n(give up)", ha="center", va="center", fontsize=7.5,
        fontweight="bold", color="#fca5a5", zorder=4, fontfamily="monospace")

# ═══════════════════════════════════════════════════════════════════════
#  DRAW EDGES
# ═══════════════════════════════════════════════════════════════════════

# START → route_question
arrow(ax, 9, 20.45, 9, 19.65, color=C["arrow"])

# route_question → compare_companies (diagonal left)
curved_arrow(ax, POS["route_question"][0]-0.5, POS["route_question"][1]-0.45,
             POS["compare_companies"][0]+0.4, POS["compare_companies"][1]+0.6,
             color=C["llm_edge"], label='"compare"', label_side="left", rad=-0.15)

# route_question → retrieve (straight down)
arrow(ax, 9, 18.75, 9, 16.95, color=C["retrieval_edge"], label='"retrieve"', label_side="right")

# route_question → direct_answer (diagonal right)
curved_arrow(ax, POS["route_question"][0]+0.5, POS["route_question"][1]-0.45,
             POS["direct_answer"][0]-0.4, POS["direct_answer"][1]+0.45,
             color=C["llm_edge"], label='"direct"', label_side="right", rad=0.15)

# compare_companies → END
curved_arrow(ax, POS["compare_companies"][0], POS["compare_companies"][1]-0.65,
             POS["END"][0]-0.3, POS["END"][1]+0.45,
             color=C["yes"], rad=0.25, label="answer ready")

# direct_answer → END
curved_arrow(ax, POS["direct_answer"][0], POS["direct_answer"][1]-0.45,
             POS["END"][0]+0.4, POS["END"][1]+0.45,
             color=C["yes"], rad=-0.25, label="answer ready")

# retrieve → grade_documents
arrow(ax, 9, 16.05, 9, 14.45, color=C["retrieval_edge"])

# grade_documents → generate (relevant=True)
arrow(ax, 9, 13.55, 9, 11.95, color=C["yes"], label="relevant ✓")

# grade_documents → rewrite_query (not relevant, gen<2)
curved_arrow(ax, POS["grade_documents"][0]+1.9, POS["grade_documents"][1],
             POS["rewrite_query"][0]-0.15, POS["rewrite_query"][1]+0.35,
             color=C["no"], rad=0.15, label="not relevant\n(gen < 2)")

# grade_documents → END_giveup (not relevant, gen>=2)
arrow(ax, POS["grade_documents"][0]-1.9, POS["grade_documents"][1],
      POS["END_giveup"][0]+0.5, POS["END_giveup"][1]+0.1,
      color="#f87171", label="not relevant\n(gen ≥ 2)", label_side="left")

# generate → check_hallucination
arrow(ax, 9, 11.05, 9, 9.45, color=C["llm_edge"])

# check_hallucination → check_usefulness (grounded=True)
arrow(ax, 9, 8.55, 9, 6.95, color=C["yes"], label="grounded ✓")

# check_hallucination → rewrite_query (not grounded, gen<2)
curved_arrow(ax, POS["check_hallucination"][0]+1.9, POS["check_hallucination"][1],
             POS["rewrite_query"][0]-0.15, POS["rewrite_query"][1]-0.15,
             color=C["no"], rad=0.0, label="not grounded\n(gen < 2)")

# check_hallucination → END (not grounded, gen>=2)
curved_arrow(ax, POS["check_hallucination"][0]-1.5, POS["check_hallucination"][1]-0.3,
             POS["END"][0]-0.5, POS["END"][1]+0.5,
             color="#f87171", rad=-0.2, label="not grounded\n(gen ≥ 2)", label_side="left")

# check_usefulness → END (useful=True)
arrow(ax, 9, 6.05, 9, 4.55, color=C["yes"], label="useful ✓")

# check_usefulness → rewrite_query (not useful, gen<2)
curved_arrow(ax, POS["check_usefulness"][0]+1.9, POS["check_usefulness"][1],
             POS["rewrite_query"][0]-0.1, POS["rewrite_query"][1]-0.65,
             color=C["no"], rad=-0.2, label="not useful\n(gen < 2)")

# check_usefulness → END (not useful, gen>=2)
curved_arrow(ax, POS["check_usefulness"][0]+0.6, POS["check_usefulness"][1]-0.45,
             POS["END"][0]+0.5, POS["END"][1]+0.45,
             color="#f87171", rad=-0.15, label="not useful\n(gen ≥ 2)")

# rewrite_query → retrieve  (long loop back up)
ax.annotate("",
    xy=(POS["retrieve"][0]+1.85, POS["retrieve"][1]),
    xytext=(POS["rewrite_query"][0], POS["rewrite_query"][1]+0.45),
    arrowprops=dict(arrowstyle="-|>", color=C["rewrite_edge"], lw=2.2,
                    mutation_scale=18,
                    connectionstyle="arc3,rad=-0.45"),
    zorder=5)
ax.text(16.3, 14, "retry\n(loop)", ha="center", va="center",
        fontsize=8, color=C["rewrite_edge"], fontweight="bold", zorder=6)

# ═══════════════════════════════════════════════════════════════════════
#  LEGEND
# ═══════════════════════════════════════════════════════════════════════
legend_x, legend_y = 0.4, 8.5
ax.text(legend_x, legend_y + 0.1, "LEGEND", fontsize=9, fontweight="bold",
        color="#94a3b8", zorder=6)
legend_items = [
    (C["router_edge"],    "Router / Grader node"),
    (C["retrieval_edge"], "Vector store retrieval"),
    (C["llm_edge"],       "LLM generation node"),
    (C["quality_edge"],   "Quality-check node"),
    (C["rewrite_edge"],   "Query rewrite node"),
    (C["yes"],            "Positive / success branch"),
    (C["no"],             "Negative / failure branch (retry if gen<2)"),
    ("#f87171",           "Hard stop (gen ≥ 2 limit reached)"),
]
for i, (color, label) in enumerate(legend_items):
    y = legend_y - 0.6 - i * 0.55
    ax.add_patch(plt.Rectangle((legend_x, y-0.15), 0.35, 0.32,
                                color=color, zorder=6))
    ax.text(legend_x + 0.55, y + 0.01, label, fontsize=7.8,
            color="#d0d0d0", va="center", zorder=6)

# ═══════════════════════════════════════════════════════════════════════
#  TITLE & SUBTITLE
# ═══════════════════════════════════════════════════════════════════════
ax.text(9, 21.65, "EarningsLens v2 — LangGraph Query Flow",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color="#f0f0f0", zorder=6)
ax.text(9, 21.2, "Adaptive RAG  •  Pinecone backend  •  Groq LLaMA-3.3-70b",
        ha="center", va="center", fontsize=9, color="#94a3b8", zorder=6)

# generation counter annotation box
gen_box = FancyBboxPatch((0.3, 0.3), 4.8, 1.2,
    boxstyle="round,pad=0.05,rounding_size=0.2",
    linewidth=1.5, edgecolor="#4b5563", facecolor="#1f2937", zorder=3)
ax.add_patch(gen_box)
ax.text(2.7, 1.1, "Loop guard: generation_count",
        ha="center", va="center", fontsize=8.5, fontweight="bold",
        color="#f59e0b", zorder=4)
ax.text(2.7, 0.65, "Increments on each generate() call.\nIf ≥ 2, all retry paths go to END instead.",
        ha="center", va="center", fontsize=7.5, color="#94a3b8", zorder=4)

# ── Save ──────────────────────────────────────────────────────────────
out = r"C:\Users\PC\Desktop\earningslens\earningslens_graph_v2.png"
plt.tight_layout(pad=0)
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print(f"Saved: {out}")
