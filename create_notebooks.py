"""
create_notebooks.py — Generates 5 Google DS Standard Jupyter Notebooks for Project 3
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(r"c:\Users\invin\OneDrive\Desktop\DataScience\rag-nlp-intelligence-platform\notebooks")


def make_cell(cell_type, source):
    if isinstance(source, str):
        lines = [line + "\n" for line in source.split("\n")]
        if lines and lines[-1] == "\n":
            lines[-1] = ""
    else:
        lines = source
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


# Notebook 1: EDA and Chunking
nb1_cells = [
    make_cell("markdown", "# 01. SQuAD 2.0 Corpus Exploratory Data Analysis & Chunking Pipeline\n\n**Author:** Avijeet Tiwari | Google Data Scientist Portfolio\n\n**Objective:** Ingest authentic Stanford Question Answering Dataset (SQuAD 2.0), analyze document lengths, sentence boundaries, answerable vs. unanswerable distribution, and establish semantic chunking boundaries."),
    make_cell("code", "import os\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nsns.set_theme(style='whitegrid')\nprint('Packages loaded successfully.')"),
    make_cell("code", "corpus_path = '../data/processed/corpus_chunks.parquet'\ngolden_path = '../data/processed/eval_golden_set.parquet'\n\ndf_corpus = pd.read_parquet(corpus_path)\ndf_golden = pd.read_parquet(golden_path)\n\nprint(f'Corpus Chunks: {len(df_corpus):,}')\nprint(f'Golden Evaluation QA Pairs: {len(df_golden):,}')"),
    make_cell("markdown", "### 1. Corpus Chunk Length Distribution\nUnderstanding token/character distribution is critical to avoid truncating retrieval passages or exceeding embedding context windows (256 tokens for MiniLM)."),
    make_cell("code", "df_corpus['char_len'] = df_corpus['text'].str.len()\ndf_corpus['word_len'] = df_corpus['text'].str.split().str.len()\n\nprint(df_corpus[['char_len', 'word_len']].describe())"),
    make_cell("code", "plt.figure(figsize=(10, 4))\nsns.histplot(df_corpus['word_len'], bins=40, kde=True, color='teal')\nplt.title('Word Count Distribution Across 4,039 Wikipedia Passages')\nplt.xlabel('Words per Passage Chunk')\nplt.ylabel('Frequency')\nplt.show()"),
    make_cell("markdown", "### 2. Answerability Balance in Golden Set\nSQuAD 2.0 uniquely tests model restraint by including unanswerable questions to detect over-confidence."),
    make_cell("code", "print(df_golden['has_answer'].value_counts(normalize=True))"),
    make_cell("code", "plt.figure(figsize=(6, 4))\nsns.countplot(data=df_golden, x='has_answer', palette='Set2')\nplt.title('SQuAD 2.0 Golden Set: Answerable vs Unanswerable QA Pairs')\nplt.xlabel('Has Valid Ground-Truth Answer')\nplt.ylabel('Count')\nplt.show()")
]

# Notebook 2: Embedding Benchmarks
nb2_cells = [
    make_cell("markdown", "# 02. Sentence-Transformers Embedding Benchmark Suite\n\n**Author:** Avijeet Tiwari | Google Data Scientist Portfolio\n\n**Objective:** Empirically benchmark candidate dense representation models (`all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`, `all-mpnet-base-v2`) on latency, memory footprint, and retrieval recall."),
    make_cell("code", "import sys, os\nsys.path.insert(0, '..')\nimport time\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.embedder import Embedder"),
    make_cell("code", "candidates = {\n    'MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',\n    'MiniLM-L12-v2': 'sentence-transformers/all-MiniLM-L12-v2'\n}\n\nsample_texts = [\n    'Deep neural networks require substantial matrix multiplication acceleration.',\n    'Photosynthesis utilizes chlorophyll pigments to absorb radiant solar energy.',\n    'The Magna Carta signed in 1215 established the principle that everyone is subject to the law.'\n] * 20\n\nbenchmark_records = []\nfor name, model_id in candidates.items():\n    emb = Embedder(model_name=model_id)\n    t0 = time.perf_counter()\n    vectors = emb.encode(sample_texts, batch_size=32)\n    dur = time.perf_counter() - t0\n    benchmark_records.append({\n        'Model': name,\n        'Embedding Dim': emb.get_embedding_dim(),\n        'Latency (s)': round(dur, 4),\n        'Throughput (texts/sec)': round(len(sample_texts) / dur, 1)\n    })\n\ndf_bench = pd.DataFrame(benchmark_records)\ndf_bench"),
    make_cell("code", "plt.figure(figsize=(8, 4))\nplt.bar(df_bench['Model'], df_bench['Throughput (texts/sec)'], color=['#4285F4', '#34A853'])\nplt.title('Embedding Inference Throughput Comparison (CPU)')\nplt.ylabel('Texts Processed per Second')\nplt.show()")
]

# Notebook 3: FAISS Vector Index
nb3_cells = [
    make_cell("markdown", "# 03. High-Performance FAISS Vector Indexing & Search Engine\n\n**Author:** Avijeet Tiwari | Google Data Scientist Portfolio\n\n**Objective:** Construct, serialize, and benchmark an inner-product dense index using Facebook AI Similarity Search (FAISS) over 4,039 passage vectors."),
    make_cell("code", "import sys, os\nsys.path.insert(0, '..')\nimport faiss\nimport numpy as np\nimport pandas as pd\nfrom src.vector_store import FAISSVectorStore\nfrom src.embedder import Embedder\nfrom src.retriever import Retriever"),
    make_cell("code", "embedder = Embedder(model_name='sentence-transformers/all-MiniLM-L6-v2')\nstore = FAISSVectorStore(embedding_dim=embedder.get_embedding_dim())\nstore.load('../models/faiss_index.index', '../models/faiss_index_meta.parquet')\nprint(f'FAISS Index loaded successfully. Total Vectors: {store.count:,}')"),
    make_cell("code", "retriever = Retriever(embedder=embedder, vector_store=store)\nquery = 'What is the function of the human cardiovascular system?'\nresults = retriever.retrieve(query, top_k=3)\n\nfor i, r in enumerate(results, 1):\n    print(f\"[{i}] Score: {r['score']:.4f} | Title: {r['title']}\")\n    print(f\"    Excerpt: {r['text'][:150]}...\\n\")")
]

# Notebook 4: RAG Generation & Gemini
nb4_cells = [
    make_cell("markdown", "# 04. Grounded LLM Generation with Gemini 1.5 Flash\n\n**Author:** Avijeet Tiwari | Google Data Scientist Portfolio\n\n**Objective:** Integrate context passages into high-grounding prompt templates with Google Gemini 1.5 Flash, measuring output faithfulness and latency."),
    make_cell("code", "import sys, os\nsys.path.insert(0, '..')\nfrom src.generator import GeminiGenerator\n\napi_key = os.environ.get('GEMINI_API_KEY', '')\nif api_key:\n    generator = GeminiGenerator(api_key=api_key)\n    print('Gemini Generator initialized successfully.')\nelse:\n    print('Please set GEMINI_API_KEY environment variable.')"),
    make_cell("code", "context_passages = [\n    {'title': 'Alexander Fleming', 'text': 'In 1928, Alexander Fleming discovered penicillin from the mould Penicillium notatum.', 'score': 0.89}\n]\n\nresult = generator.generate(\n    question='Who discovered penicillin and in which year?',\n    retrieved_passages=context_passages\n)\n\nprint(f\"Answer: {result['answer']}\")\nprint(f\"Latency: {result['latency_ms']} ms\")\nprint(f\"Sources: {result['sources']}\")")
]

# Notebook 5: Hallucination & Evaluation
nb5_cells = [
    make_cell("markdown", "# 05. Hallucination Detection & Comprehensive RAG Evaluation\n\n**Author:** Avijeet Tiwari | Google Data Scientist Portfolio\n\n**Objective:** Audit answer faithfulness, compute ROUGE-L, Named-Entity overlap, and detect model hallucinations on SQuAD 2.0 evaluation data."),
    make_cell("code", "import sys, os\nsys.path.insert(0, '..')\nfrom src.evaluator import RAGEvaluator\nfrom src.hallucination_detector import HallucinationDetector\nimport pandas as pd"),
    make_cell("code", "evaluator = RAGEvaluator(use_bertscore=False)\ndetector = HallucinationDetector(use_semantic=False)\n\ncontext = [{'title': 'Moon Landing', 'text': 'Apollo 11 landed astronauts Neil Armstrong and Buzz Aldrin on the Moon on July 20, 1969.'}]\n\n# Test Faithful Answer\nres_faithful = detector.detect('Neil Armstrong and Buzz Aldrin landed on the Moon in July 1969.', context)\nprint('Faithful Case:', res_faithful['verdict'], '| Score:', res_faithful['composite_score'])\n\n# Test Hallucinated Answer\nres_halluc = detector.detect('Apollo 11 landed on Mars with crew member Yuri Gagarin in 1985.', context)\nprint('Hallucinated Case:', res_halluc['verdict'], '| Score:', res_halluc['composite_score'])\nprint('Flagged Claims:', res_halluc['flagged_claims'])")
]

notebooks = [
    ("01_dataset_eda_and_chunking.ipynb", nb1_cells),
    ("02_embedding_benchmarks.ipynb", nb2_cells),
    ("03_faiss_vector_index.ipynb", nb3_cells),
    ("04_rag_generation_and_gemini.ipynb", nb4_cells),
    ("05_hallucination_and_ragas_evaluation.ipynb", nb5_cells),
]

for filename, cells in notebooks:
    nb_json = make_notebook(cells)
    target_path = NOTEBOOKS_DIR / filename
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2)
    print(f"Generated {filename}")

print("All 5 notebooks generated successfully.")
