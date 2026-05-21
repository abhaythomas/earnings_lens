"""
pages/RAG_Performance.py — RAG Evaluation Dashboard

Shows:
- Live metrics from query_log.jsonl (MRR, latency, route distribution)
- RAGAS scores from eval/scores_*.json (faithfulness, answer relevancy,
  context recall, context precision)
- Per-question breakdown with best and worst performing examples
"""

import os
import json
import glob
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Performance · EarningsLens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS (minimal, consistent with main app) ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #64748b !important; }
hr { border-color: rgba(255,255,255,0.05) !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    margin=dict(l=10, r=10, t=30, b=10),
)

# ── Data loaders ──────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_query_log() -> list[dict]:
    log_file = os.path.join("logs", "query_log.jsonl")
    if not os.path.exists(log_file):
        return []
    entries = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


@st.cache_data(ttl=60)
def load_latest_ragas_scores() -> dict | None:
    """Load the most recent scores_*.json from the eval directory."""
    pattern = os.path.join("eval", "scores_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_eval_dataset() -> list[dict]:
    path = os.path.join("eval", "eval_dataset.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Compute live metrics from query log ───────────────────────────────

def compute_live_metrics(entries: list[dict]) -> dict:
    if not entries:
        return {}

    latencies = [e["latency_ms"] for e in entries if e.get("latency_ms")]
    grounded  = [e["is_grounded"] for e in entries if e.get("is_grounded") is not None]
    routes    = [e.get("route", "unknown") for e in entries]

    # MRR from grade_vector
    rr_scores = []
    for e in entries:
        gv = e.get("grade_vector")
        if not gv:
            continue
        for rank, grade in enumerate(gv, start=1):
            if grade == 1:
                rr_scores.append(1 / rank)
                break
        else:
            rr_scores.append(0)

    route_counts = {}
    for r in routes:
        route_counts[r] = route_counts.get(r, 0) + 1

    return {
        "total_queries": len(entries),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else None,
        "grounded_pct": round(100 * sum(grounded) / len(grounded), 1) if grounded else None,
        "mrr": round(sum(rr_scores) / len(rr_scores), 3) if rr_scores else None,
        "mrr_n": len(rr_scores),
        "route_counts": route_counts,
        "latencies": latencies,
        "timestamps": [e.get("timestamp", "") for e in entries],
    }


# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
    <h1 style="font-size:1.6rem;font-weight:700;letter-spacing:-0.03em;color:#f1f5f9;margin:0 0 4px 0;">
        RAG Performance Stats
    </h1>
    <p style="color:#64748b;font-size:0.88rem;margin:0;">
        Live query metrics + offline RAGAS evaluation scores
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────
log_entries    = load_query_log()
ragas_data     = load_latest_ragas_scores()
eval_dataset   = load_eval_dataset()
live           = compute_live_metrics(log_entries)

if st.button("↺  Refresh", type="secondary"):
    load_query_log.clear()
    load_latest_ragas_scores.clear()
    st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — Live Query Metrics
# ══════════════════════════════════════════════════════════════════════
st.markdown("### Live Query Metrics")
st.caption("Computed from every query logged since the app started.")

if not live:
    st.info("No queries logged yet. Ask something in the main app first.")
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Queries", live["total_queries"])
    c2.metric("Avg Latency", f"{live['avg_latency_ms']}ms" if live.get("avg_latency_ms") else "—")
    c3.metric("P95 Latency", f"{live['p95_latency_ms']}ms" if live.get("p95_latency_ms") else "—")
    c4.metric("Grounded %",
              f"{live['grounded_pct']}%" if live.get("grounded_pct") is not None else "—")
    c5.metric("MRR",
              f"{live['mrr']}" if live.get("mrr") else "—",
              help=f"Mean Reciprocal Rank across {live.get('mrr_n', 0)} retrieve-route queries. 1.0 = perfect.")

    col_left, col_right = st.columns(2)

    # Route distribution pie
    with col_left:
        route_counts = live.get("route_counts", {})
        if route_counts:
            fig_pie = go.Figure(go.Pie(
                labels=list(route_counts.keys()),
                values=list(route_counts.values()),
                hole=0.55,
                marker_colors=["#6366f1", "#0ea5e9", "#10b981"],
                textinfo="label+percent",
                textfont=dict(size=12),
            ))
            fig_pie.update_layout(
                title="Route Distribution",
                showlegend=False,
                height=280,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Latency histogram
    with col_right:
        latencies = live.get("latencies", [])
        if latencies:
            fig_lat = go.Figure(go.Histogram(
                x=latencies,
                nbinsx=20,
                marker_color="#6366f1",
                opacity=0.8,
            ))
            fig_lat.update_layout(
                title="Latency Distribution (ms)",
                xaxis_title="ms",
                yaxis_title="queries",
                height=280,
                **PLOTLY_LAYOUT,
            )
            fig_lat.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
            fig_lat.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
            st.plotly_chart(fig_lat, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — RAGAS Evaluation Scores
# ══════════════════════════════════════════════════════════════════════
st.markdown("### RAGAS Offline Evaluation")

METRIC_META = {
    "faithfulness":       ("Faithfulness",        "#6366f1", "Are all claims in the answer supported by retrieved docs?"),
    "answer_relevancy":   ("Answer Relevancy",     "#0ea5e9", "Does the answer actually address the question?"),
    "context_recall":     ("Context Recall",       "#10b981", "Did retrieval surface the chunks needed to answer?"),
    "context_precision":  ("Context Precision",    "#f59e0b", "Of what was retrieved, how much was actually needed?"),
}

if ragas_data is None:
    st.info(
        "No RAGAS scores found yet. Run the eval first:\n\n"
        "```bash\nvenv2/Scripts/python.exe eval_ragas.py --run\n```"
    )
else:
    agg = ragas_data.get("aggregate_scores", {})
    run_date = ragas_data.get("run_at", "")[:10]
    n_q = ragas_data.get("n_questions", 0)

    st.caption(f"Last run: **{run_date}** · {n_q} questions evaluated · judge: llama-3.3-70b-versatile")

    # Metric cards
    cols = st.columns(4)
    for col, (key, (label, color, tip)) in zip(cols, METRIC_META.items()):
        val = agg.get(key)
        col.metric(label, f"{val:.3f}" if val is not None else "—", help=tip)

    col_bar, col_radar = st.columns([3, 2])

    # Horizontal bar chart
    with col_bar:
        keys   = [k for k in METRIC_META if k in agg]
        labels = [METRIC_META[k][0] for k in keys]
        values = [agg[k] for k in keys]
        colors = [METRIC_META[k][1] for k in keys]

        fig_bar = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_bar.update_layout(
            title="Aggregate RAGAS Scores  (0–1, higher is better)",
            xaxis=dict(range=[0, 1.15], gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            height=280,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Radar chart
    with col_radar:
        radar_labels = [METRIC_META[k][0] for k in METRIC_META if k in agg]
        radar_values = [agg[k] for k in METRIC_META if k in agg]
        radar_labels_closed = radar_labels + [radar_labels[0]]
        radar_values_closed = radar_values + [radar_values[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_values_closed,
            theta=radar_labels_closed,
            fill="toself",
            fillcolor="rgba(99,102,241,0.15)",
            line=dict(color="#6366f1", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="rgba(255,255,255,0.08)",
                                tickfont=dict(size=9)),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            ),
            title="Score Radar",
            height=280,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Per-question score distribution
    per_row = ragas_data.get("per_row", [])
    if per_row:
        st.markdown("#### Per-Question Score Distribution")

        all_scores = []
        for row in per_row:
            scores = row.get("scores", {})
            avg = None
            vals = [v for v in scores.values() if v is not None]
            if vals:
                avg = round(sum(vals) / len(vals), 3)
            all_scores.append({
                "question": row["question"][:60] + "…",
                "avg_score": avg,
                **{k: scores.get(k) for k in METRIC_META},
            })
        all_scores.sort(key=lambda x: x["avg_score"] or 0, reverse=True)

        # Stacked bar per question
        fig_stack = go.Figure()
        for key, (label, color, _) in METRIC_META.items():
            fig_stack.add_trace(go.Bar(
                name=label,
                x=[r["question"] for r in all_scores],
                y=[r.get(key) or 0 for r in all_scores],
                marker_color=color,
                opacity=0.85,
            ))
        fig_stack.update_layout(
            barmode="group",
            xaxis=dict(tickangle=-35, tickfont=dict(size=9),
                       gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", y=1.08),
            height=360,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3 — Best & worst cases
    # ══════════════════════════════════════════════════════════════════
    st.markdown("### Sample Cases")
    tab_good, tab_bad = st.tabs(["✅  Where RAG worked well", "❌  Where RAG struggled"])

    scored = [
        r for r in per_row
        if r.get("scores") and any(v is not None for v in r["scores"].values())
    ]

    def avg_score(row):
        vals = [v for v in row["scores"].values() if v is not None]
        return sum(vals) / len(vals) if vals else 0

    best  = sorted(scored, key=avg_score, reverse=True)[:5]
    worst = sorted(scored, key=avg_score)[:5]

    def score_badge(val, key):
        color = "#10b981" if val and val >= 0.7 else "#f59e0b" if val and val >= 0.4 else "#ef4444"
        label = METRIC_META.get(key, (key,))[0]
        return (f'<span style="display:inline-block;background:rgba(0,0,0,0.2);'
                f'border:1px solid {color}44;color:{color};'
                f'padding:2px 8px;border-radius:12px;font-size:11px;margin:2px;">'
                f'{label}: {val:.2f}</span>') if val is not None else ""

    def render_cases(cases):
        # Find matching ground truth from eval dataset
        gt_map = {row["question"]: row for row in eval_dataset}

        for i, row in enumerate(cases):
            q = row["question"]
            scores = row["scores"]
            avg = avg_score(row)
            route = row.get("route", "")
            is_grounded = row.get("is_grounded")

            badges = " ".join(score_badge(scores.get(k), k) for k in METRIC_META)

            st.markdown(
                f'<div style="border:1px solid rgba(255,255,255,0.07);border-radius:12px;'
                f'padding:1rem 1.2rem;margin-bottom:1rem;background:rgba(255,255,255,0.02);">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">'
                f'<span style="font-size:0.82rem;color:#64748b;">#{i+1} · route: {route} · '
                f'grounded: {"✅" if is_grounded else "❌" if is_grounded is False else "—"}</span>'
                f'<span style="font-size:1rem;font-weight:700;color:#f1f5f9;">avg {avg:.2f}</span>'
                f'</div>'
                f'<p style="font-weight:600;color:#f1f5f9;margin:0 0 0.5rem 0;">{q}</p>'
                f'<div style="margin-bottom:0.6rem;">{badges}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Show ground truth if available
            if q in gt_map:
                with st.expander("Ground truth"):
                    st.markdown(gt_map[q]["ground_truth"])

    with tab_good:
        if best:
            render_cases(best)
        else:
            st.info("No scored rows yet.")

    with tab_bad:
        if worst:
            render_cases(worst)
            st.markdown("""
            <div style="border-left:3px solid #f59e0b;padding:0.75rem 1rem;
                        background:rgba(245,158,11,0.06);border-radius:0 8px 8px 0;margin-top:1rem;">
            <strong style="color:#f59e0b;">Common failure patterns:</strong>
            <ul style="color:#94a3b8;margin:0.5rem 0 0 0;font-size:0.85rem;">
            <li><strong>Low Context Recall</strong> — the right chunk wasn't retrieved.
                Try increasing k (currently k=5) or improving chunk boundaries.</li>
            <li><strong>Low Faithfulness</strong> — the LLM added claims not in the sources.
                Often happens when retrieved context is too sparse.</li>
            <li><strong>Low Answer Relevancy</strong> — the question was too vague or the router
                sent it down the wrong path (check route column above).</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No scored rows yet.")

st.divider()

# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — How to improve
# ══════════════════════════════════════════════════════════════════════
with st.expander("How to run / re-run the evaluation"):
    st.markdown("""
```bash
# Generate a new 20-question synthetic dataset from Pinecone chunks
venv2/Scripts/python.exe eval_ragas.py --generate --n 20

# Score the dataset (requires ~50k Groq tokens — run at start of day)
venv2/Scripts/python.exe eval_ragas.py --run

# Both at once
venv2/Scripts/python.exe eval_ragas.py --generate --run --n 20
```

**Metric guide:**

| Metric | What it measures | Target |
|---|---|---|
| Faithfulness | All answer claims supported by retrieved docs | > 0.85 |
| Answer Relevancy | Answer addresses the question asked | > 0.80 |
| Context Recall | Retrieved chunks contain the needed info | > 0.75 |
| Context Precision | Retrieved chunks are all actually needed | > 0.70 |
| MRR (live) | Relevant doc appears early in top-5 results | > 0.60 |
    """)
