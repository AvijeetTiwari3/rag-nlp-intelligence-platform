"""
streamlit_app.py — RAG NLP Intelligence Platform Dashboard
============================================================
Interactive Streamlit dashboard for the Production-Grade RAG system.

Modules:
  1. Live RAG Query Interface
  2. Retrieval Quality Explorer
  3. Hallucination Detection Lab
  4. Evaluation Metrics Dashboard
  5. Corpus & Embedding Explorer
  6. System Health Monitor
  7. Model Card & Architecture

Author : Avijeet Tiwari | Google Data Scientist Portfolio
Dataset: SQuAD 2.0 Wikipedia corpus (4,039 chunks, 130K+ QA pairs)
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.WARNING)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG NLP Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white; text-align: center;
  }
  .metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem; border-radius: 10px; color: white; text-align: center; margin: 0.3rem 0;
  }
  .verdict-faithful  { background:#e8f5e9; border-left:5px solid #4caf50; padding:0.8rem; border-radius:6px; margin:0.5rem 0; }
  .verdict-uncertain { background:#fff8e1; border-left:5px solid #ff9800; padding:0.8rem; border-radius:6px; margin:0.5rem 0; }
  .verdict-hallucinated { background:#ffebee; border-left:5px solid #f44336; padding:0.8rem; border-radius:6px; margin:0.5rem 0; }
  .answer-box { background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:1rem; font-size:1rem; }
  .source-chip { display:inline-block; background:#e3f2fd; color:#1565c0; padding:0.2rem 0.6rem;
                 border-radius:12px; font-size:0.8rem; margin:0.2rem; }
</style>
""", unsafe_allow_html=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or (st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else "")

# ── Cached Resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG Pipeline (first load ~60s)...")
def load_pipeline():
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline(api_key=GEMINI_API_KEY)


@st.cache_data(show_spinner=False)
def load_corpus():
    path = BASE_DIR / "data" / "processed" / "corpus_chunks.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_golden_set():
    path = BASE_DIR / "data" / "processed" / "eval_golden_set.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAG NLP Platform")
    st.markdown("**Google Data Scientist Portfolio**")
    st.markdown("---")

    module = st.radio(
        "Navigation Modules",
        [
            "1. Live RAG Query",
            "2. Retrieval Explorer",
            "3. Hallucination Lab",
            "4. Evaluation Metrics",
            "5. Corpus Explorer",
            "6. System Health",
            "7. Model Card",
        ],
        index=0,
    )
    st.markdown("---")
    st.markdown("""
    **Dataset:** SQuAD 2.0 (130K+ QA pairs)  
    **Corpus:** 4,039 Wikipedia chunks  
    **Embedder:** MiniLM-L6-v2 (384-dim)  
    **Generator:** Gemini 3.5 Flash  
    **Vector DB:** FAISS IndexFlatIP  
    """)
    st.markdown("---")
    st.markdown("Stack: Python · SentenceTransformers · FAISS · Gemini · FastAPI · Streamlit")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: Live RAG Query
# ═══════════════════════════════════════════════════════════════════════════════
if module == "1. Live RAG Query":
    st.markdown("""
    <div class="main-header">
      <h1>🧠 Live RAG Query Interface</h1>
      <p>Ask any question answered by the SQuAD 2.0 Wikipedia corpus</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g. Who invented the telephone? What is photosynthesis?",
            key="rag_question",
        )
    with col2:
        top_k = st.slider("Top-K passages", 1, 10, 5)

    example_questions = [
        "Who invented the telephone?",
        "What is the speed of light?",
        "Who wrote Romeo and Juliet?",
        "What causes earthquakes?",
        "When was the Eiffel Tower built?",
    ]
    st.markdown("**Quick examples:**")
    cols = st.columns(len(example_questions))
    for i, eq in enumerate(example_questions):
        if cols[i].button(eq[:25] + "...", key=f"eq_{i}"):
            question = eq

    if st.button("🔍 Search & Generate Answer", type="primary") and question:
        with st.spinner("Retrieving passages and generating answer..."):
            try:
                pipeline = load_pipeline()
                t0 = time.perf_counter()
                result = pipeline.query(question, top_k=top_k)
                total_time = (time.perf_counter() - t0) * 1000

                # Answer section
                st.markdown("### Generated Answer")
                verdict = result.get("hallucination_verdict", "N/A")
                verdict_class = {
                    "FAITHFUL": "verdict-faithful",
                    "UNCERTAIN": "verdict-uncertain",
                    "HALLUCINATED": "verdict-hallucinated",
                }.get(verdict, "answer-box")
                verdict_icon = {"FAITHFUL": "✅", "UNCERTAIN": "⚠️", "HALLUCINATED": "🚨"}.get(verdict, "ℹ️")

                st.markdown(f'<div class="{verdict_class}"><b>{verdict_icon} Faithfulness: {verdict}</b></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

                # Sources
                if result.get("sources"):
                    st.markdown("**Sources:**")
                    chips = " ".join(f'<span class="source-chip">{s[:40]}</span>' for s in result["sources"])
                    st.markdown(chips, unsafe_allow_html=True)

                # Metrics
                st.markdown("### Latency Breakdown")
                lat = result.get("stage_latencies", {})
                lat_df = pd.DataFrame(
                    [(k.replace("_ms", "").title(), v) for k, v in lat.items()],
                    columns=["Stage", "Latency (ms)"],
                )
                fig = px.bar(lat_df, x="Stage", y="Latency (ms)", color="Latency (ms)",
                             color_continuous_scale="Blues", title="Pipeline Stage Latencies")
                st.plotly_chart(fig, use_container_width=True)

                # Retrieval scores
                scores = result.get("retrieval_scores", [])
                if scores:
                    score_df = pd.DataFrame(
                        {"Passage": [f"P{i+1}" for i in range(len(scores))], "Similarity": scores}
                    )
                    fig2 = px.bar(score_df, x="Passage", y="Similarity",
                                  title="Retrieval Similarity Scores", color="Similarity",
                                  color_continuous_scale="Viridis")
                    st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Tip: Make sure the FAISS index is built. Run `python src/data_pipeline.py` then `python src/embedder.py` first.")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: Retrieval Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "2. Retrieval Explorer":
    st.markdown("""
    <div class="main-header">
      <h1>🔍 Retrieval Quality Explorer</h1>
      <p>Inspect retrieved passages and similarity scores for any query</p>
    </div>
    """, unsafe_allow_html=True)

    corpus_df = load_corpus()
    if corpus_df.empty:
        st.warning("Corpus not found. Run data_pipeline.py first.")
    else:
        st.metric("Corpus Size", f"{len(corpus_df):,} chunks")

        col1, col2 = st.columns([3, 1])
        with col1:
            r_question = st.text_input("Query for retrieval:", "What is machine learning?")
        with col2:
            r_k = st.slider("k", 1, 20, 5, key="rk")

        if st.button("Retrieve Passages", type="primary"):
            try:
                pipeline = load_pipeline()
                retrieved = pipeline.retriever.retrieve(r_question, top_k=r_k)

                st.markdown(f"### Top-{r_k} Retrieved Passages")
                for i, p in enumerate(retrieved, 1):
                    with st.expander(f"Passage {i}: {p.get('title','?')} (score={p.get('score',0):.4f})"):
                        st.write(p.get("text", ""))

                # Score distribution
                scores = [p.get("score", 0) for p in retrieved]
                fig = go.Figure(go.Bar(
                    x=[f"P{i+1}" for i in range(len(scores))],
                    y=scores,
                    marker_color=px.colors.sequential.Plasma,
                ))
                fig.update_layout(title="Retrieval Similarity Scores", xaxis_title="Passage", yaxis_title="Cosine Similarity")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Retrieval error: {e}")

        # Corpus stats
        st.markdown("---")
        st.markdown("### Corpus Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Chunks", f"{len(corpus_df):,}")
        col2.metric("Unique Topics", f"{corpus_df['title'].nunique():,}" if "title" in corpus_df.columns else "N/A")
        if "text" in corpus_df.columns:
            avg_len = corpus_df["text"].str.len().mean()
            col3.metric("Avg Chunk Length", f"{avg_len:.0f} chars")

        if "title" in corpus_df.columns:
            top_topics = corpus_df["title"].value_counts().head(20).reset_index()
            top_topics.columns = ["Topic", "Chunks"]
            fig2 = px.bar(top_topics, x="Chunks", y="Topic", orientation="h",
                          title="Top 20 Topics by Chunk Count", color="Chunks",
                          color_continuous_scale="Teal")
            st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: Hallucination Lab
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "3. Hallucination Lab":
    st.markdown("""
    <div class="main-header">
      <h1>🚨 Hallucination Detection Lab</h1>
      <p>Audit any generated answer for faithfulness to source context</p>
    </div>
    """, unsafe_allow_html=True)

    from src.hallucination_detector import HallucinationDetector
    detector = HallucinationDetector(use_semantic=False)

    c1, c2 = st.columns(2)
    with c1:
        h_answer = st.text_area("Generated Answer:", "The Eiffel Tower was built in 1889 by Gustave Eiffel.", height=100)
    with c2:
        h_context = st.text_area("Source Context:", "The Eiffel Tower is located in Paris, France. It was constructed from 1887 to 1889 by Gustave Eiffel.", height=100)

    if st.button("Detect Hallucination", type="primary"):
        result = detector.detect(h_answer, [{"text": h_context, "title": "Context"}])

        verdict = result["verdict"]
        verdict_icon = {"FAITHFUL": "✅", "UNCERTAIN": "⚠️", "HALLUCINATED": "🚨"}.get(verdict, "ℹ️")
        verdict_color = {"FAITHFUL": "green", "UNCERTAIN": "orange", "HALLUCINATED": "red"}.get(verdict, "blue")

        st.markdown(f"## {verdict_icon} Verdict: :{verdict_color}[**{verdict}**]")
        st.markdown(f"**Composite Faithfulness Score:** `{result['composite_score']:.4f}`")

        col1, col2, col3 = st.columns(3)
        col1.metric("Token Faithfulness", f"{result['token_faithfulness']:.4f}")
        col2.metric("Entity Coverage", f"{result['entity_score']:.4f}")
        col3.metric("Numeric Coverage", f"{result['numeric_score']:.4f}")

        # Radar chart
        categories = ["Token Faithfulness", "Entity Coverage", "Numeric Coverage"]
        values = [result["token_faithfulness"], result["entity_score"], result["numeric_score"]]
        fig = go.Figure(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]],
                                        fill="toself", name="Faithfulness Scores",
                                        line_color=verdict_color))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          title="Faithfulness Radar")
        st.plotly_chart(fig, use_container_width=True)

        if result["flagged_claims"]:
            st.markdown("### Flagged Unsupported Claims")
            for claim in result["flagged_claims"]:
                st.markdown(f"- _{claim}_")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "4. Evaluation Metrics":
    st.markdown("""
    <div class="main-header">
      <h1>📊 RAG Evaluation Metrics Dashboard</h1>
      <p>ROUGE-L · BERTScore · Faithfulness · Composite Quality on SQuAD 2.0</p>
    </div>
    """, unsafe_allow_html=True)

    from src.evaluator import RAGEvaluator
    evaluator = RAGEvaluator(use_bertscore=False)

    st.markdown("### Quick Single-Pair Evaluation")
    c1, c2 = st.columns(2)
    with c1:
        eval_gen = st.text_area("Generated Answer:", "Paris is the capital of France.", height=80)
        eval_ref = st.text_area("Reference Answer:", "The capital of France is Paris.", height=80)
    with c2:
        eval_ctx = st.text_area("Context:", "Paris is the capital and most populous city of France.", height=80)

    if st.button("Evaluate", type="primary"):
        result = evaluator.evaluate_single(
            question="What is the capital of France?",
            generated_answer=eval_gen,
            reference_answer=eval_ref,
            context_passages=[{"text": eval_ctx, "title": "France"}],
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROUGE-L", f"{result['rouge_l']:.4f}")
        col2.metric("Faithfulness", f"{result['faithfulness']:.4f}")
        col3.metric("Composite", f"{result['composite_score']:.4f}")
        col4.metric("Hallucinated", "Yes" if result["hallucinated"] else "No")

        metrics_df = pd.DataFrame([{
            "Metric": k,
            "Score": v
        } for k, v in result.items() if isinstance(v, float)])
        fig = px.bar(metrics_df, x="Metric", y="Score", color="Score",
                     color_continuous_scale="RdYlGn", title="Evaluation Scores", range_y=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # Pre-computed benchmark results if available
    bench_path = BASE_DIR / "reports" / "benchmark_results.parquet"
    if bench_path.exists():
        st.markdown("---")
        st.markdown("### Embedding Model Benchmark Results")
        bench_df = pd.read_parquet(bench_path)
        st.dataframe(bench_df, use_container_width=True)

        fig2 = px.bar(bench_df, x="model", y=["precision_at_1", "recall_at_5", "mrr", "ndcg_at_5"],
                      barmode="group", title="Retrieval Metrics by Embedding Model")
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5: Corpus Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "5. Corpus Explorer":
    st.markdown("""
    <div class="main-header">
      <h1>📚 Corpus & Embedding Explorer</h1>
      <p>Explore the SQuAD 2.0 Wikipedia passage corpus</p>
    </div>
    """, unsafe_allow_html=True)

    corpus_df = load_corpus()
    golden_df = load_golden_set()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Corpus Chunks", f"{len(corpus_df):,}" if not corpus_df.empty else "N/A")
    col2.metric("Golden QA Pairs", f"{len(golden_df):,}" if not golden_df.empty else "N/A")
    col3.metric("Embedding Dim", "384 (MiniLM-L6)")
    col4.metric("Index Type", "FAISS IndexFlatIP")

    if not corpus_df.empty:
        st.markdown("### Corpus Sample")
        st.dataframe(corpus_df.head(50), use_container_width=True)

        if "text" in corpus_df.columns:
            lengths = corpus_df["text"].str.len()
            fig = px.histogram(lengths, nbins=50, title="Chunk Length Distribution (characters)",
                               labels={"value": "Length (chars)", "count": "Frequency"},
                               color_discrete_sequence=["#667eea"])
            st.plotly_chart(fig, use_container_width=True)

    if not golden_df.empty:
        st.markdown("### Golden Evaluation Set")
        answerable = golden_df["has_answer"].sum() if "has_answer" in golden_df.columns else 0
        unanswerable = len(golden_df) - answerable
        fig2 = px.pie(
            values=[answerable, unanswerable],
            names=["Answerable", "Unanswerable"],
            title="Golden Set: Answerable vs Unanswerable Questions",
            color_discrete_sequence=["#4caf50", "#f44336"],
        )
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6: System Health
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "6. System Health":
    st.markdown("""
    <div class="main-header">
      <h1>💚 System Health Monitor</h1>
      <p>Real-time status of all RAG pipeline components</p>
    </div>
    """, unsafe_allow_html=True)

    checks = {}

    # Corpus
    corpus_path = BASE_DIR / "data" / "processed" / "corpus_chunks.parquet"
    checks["Corpus (corpus_chunks.parquet)"] = corpus_path.exists()

    # Golden set
    golden_path = BASE_DIR / "data" / "processed" / "eval_golden_set.parquet"
    checks["Golden Set (eval_golden_set.parquet)"] = golden_path.exists()

    # FAISS index
    faiss_path = BASE_DIR / "models" / "faiss_index.index"
    checks["FAISS Index (faiss_index.index)"] = faiss_path.exists()

    # FAISS meta
    meta_path = BASE_DIR / "models" / "faiss_index_meta.parquet"
    checks["FAISS Meta (faiss_index_meta.parquet)"] = meta_path.exists()

    # SQuAD raw
    squad_path = BASE_DIR / "data" / "raw" / "squad_v2_train.parquet"
    checks["SQuAD 2.0 Train (raw)"] = squad_path.exists()

    # Source files
    for src_file in ["embedder.py", "retriever.py", "generator.py", "rag_pipeline.py", "evaluator.py", "hallucination_detector.py"]:
        checks[f"src/{src_file}"] = (BASE_DIR / "src" / src_file).exists()

    st.markdown("### Component Status")
    for name, status in checks.items():
        icon = "✅" if status else "❌"
        color = "green" if status else "red"
        st.markdown(f":{color}[{icon}] **{name}**")

    passed = sum(checks.values())
    total = len(checks)
    health_pct = int(passed / total * 100)
    st.markdown(f"### Overall Health: **{health_pct}%** ({passed}/{total} components OK)")
    st.progress(health_pct / 100)

    st.markdown("---")
    st.markdown("### File Sizes")
    file_sizes = []
    for name, ok in checks.items():
        # Find the path
        for p in [corpus_path, golden_path, faiss_path, meta_path, squad_path]:
            if p.exists():
                size_mb = p.stat().st_size / (1024**2)
                file_sizes.append({"File": p.name, "Size (MB)": round(size_mb, 2)})
    if file_sizes:
        size_df = pd.DataFrame(file_sizes).drop_duplicates("File")
        fig = px.bar(size_df, x="File", y="Size (MB)", color="Size (MB)",
                     title="Data File Sizes", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 7: Model Card
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "7. Model Card":
    st.markdown("""
    <div class="main-header">
      <h1>📋 Model Card & System Architecture</h1>
      <p>Production documentation for the RAG NLP Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## System Overview

    The **RAG NLP Intelligence Platform** is a production-grade Retrieval-Augmented Generation system
    built on real-world Wikipedia passages from SQuAD 2.0.

    ## Architecture

    ```
    User Query
        │
        ▼
    [Query Encoder] — sentence-transformers/all-MiniLM-L6-v2 (384-dim)
        │
        ▼
    [FAISS Vector Store] — IndexFlatIP, 4,039 Wikipedia chunks
        │  Top-K cosine similarity search
        ▼
    [Context Assembly] — Passage ranking + citation injection
        │
        ▼
    [Gemini 1.5 Flash] — Grounded generation with system prompt
        │
        ▼
    [Hallucination Detector] — Token overlap + Named Entity + Numeric verification
        │
        ▼
    Structured Response (answer + verdict + sources + latency)
    ```

    ## Dataset

    | Property | Value |
    |----------|-------|
    | Dataset | SQuAD 2.0 (Stanford QA Dataset) |
    | Source | Wikipedia articles |
    | Train QA pairs | 130,319 |
    | Val QA pairs | 11,873 |
    | Corpus chunks | 4,039 |
    | Eval golden set | 1,000 (800 answerable + 200 unanswerable) |
    | Answerable % | 80% |
    | Unanswerable % | 20% |

    ## Embedding Model

    | Property | Value |
    |----------|-------|
    | Model | all-MiniLM-L6-v2 |
    | Embedding Dimensions | 384 |
    | Max Sequence Length | 256 tokens |
    | Encoding Throughput | ~60 chunks/sec |
    | FAISS Index Type | IndexFlatIP (inner product) |
    | Similarity Metric | Cosine similarity |

    ## Generation Model

    | Property | Value |
    |----------|-------|
    | Model | Gemini 1.5 Flash |
    | Provider | Google AI Studio (free tier) |
    | Temperature | 0.1 (factual mode) |
    | Max Output Tokens | 512 |
    | Prompt Strategy | RAG-grounded with citation injection |
    | Unanswerable Handling | Explicit refusal when context insufficient |

    ## Evaluation Metrics

    | Metric | Description | Target |
    |--------|-------------|--------|
    | ROUGE-L F1 | Lexical overlap with reference | > 0.40 |
    | BERTScore F1 | Semantic similarity | > 0.65 |
    | Faithfulness | Token overlap answer vs context | > 0.50 |
    | Hallucination Rate | % answers HALLUCINATED | < 15% |
    | Composite Score | Weighted average | > 0.55 |

    ## Intended Use

    - Educational: Understanding RAG pipeline architecture
    - Research: Baseline for QA systems on Wikipedia passages
    - Portfolio: Demonstrating production ML engineering skills

    ## Limitations

    - Corpus limited to SQuAD 2.0 Wikipedia passages (not the full web)
    - Hallucination detector uses heuristics, not a trained NLI model
    - Free-tier Gemini API has rate limits (~60 requests/minute)

    ## Author

    **Avijeet Tiwari** | Data Scientist  
    Google Data Scientist Portfolio Project #3  
    """)
