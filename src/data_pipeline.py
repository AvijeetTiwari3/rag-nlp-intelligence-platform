"""
Corpus Chunking & Golden Evaluation Set Construction Engine
Processes Stanford SQuAD 2.0 Wikipedia passages into retrieval-ready segments.
"""

import os
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional


@dataclass
class CorpusChunk:
    chunk_id: str
    source_title: str
    text: str
    word_count: int


def build_rag_corpus(
    raw_squad_train_path: str,
    raw_squad_val_path: str,
    output_dir: str = "data/processed",
    chunk_size: int = 250,
    chunk_overlap: int = 40,
    max_passages: int = 4000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build chunked document corpus and golden evaluation benchmark dataset.
    """
    if not os.path.exists(raw_squad_train_path):
        raise FileNotFoundError(f"Raw SQuAD train file not found at: {raw_squad_train_path}")

    print("Loading SQuAD 2.0 train and validation parquet files...")
    train_df = pd.read_parquet(raw_squad_train_path)
    val_df = pd.read_parquet(raw_squad_val_path)

    # 1. Deduplicate unique passages from train set
    unique_passages = train_df[["title", "context"]].drop_duplicates(subset=["context"]).reset_index(drop=True)
    if max_passages and len(unique_passages) > max_passages:
        unique_passages = unique_passages.iloc[:max_passages]

    print(f"Processing {len(unique_passages)} unique Wikipedia passages into chunks...")

    chunks = []
    chunk_idx = 0
    for _, row in unique_passages.iterrows():
        title = str(row["title"])
        text = str(row["context"])
        words = text.split()

        step = max(1, chunk_size - chunk_overlap)
        for start in range(0, len(words), step):
            segment = words[start : start + chunk_size]
            if len(segment) < 25:  # skip tiny trailing fragments
                continue
            chunk_id = f"chunk_{chunk_idx:05d}"
            chunks.append(CorpusChunk(
                chunk_id=chunk_id,
                source_title=title,
                text=" ".join(segment),
                word_count=len(segment)
            ))
            chunk_idx += 1

    corpus_df = pd.DataFrame([asdict(c) for c in chunks])
    print(f"Generated {len(corpus_df)} retrieval chunks.")

    # 2. Build Golden Evaluation Set (800 answerable + 200 unanswerable QA pairs)
    # Answerable: answers['text'] has at least one element
    def has_answer(ans):
        if ans is None:
            return False
        if isinstance(ans, dict) and "text" in ans:
            return len(ans["text"]) > 0
        return False

    val_df["is_answerable"] = val_df["answers"].apply(has_answer)
    answerable_pool = val_df[val_df["is_answerable"]].copy()
    unanswerable_pool = val_df[~val_df["is_answerable"]].copy()

    # Extract ground truth string
    def extract_gt(ans):
        if isinstance(ans, dict) and "text" in ans and len(ans["text"]) > 0:
            return str(ans["text"][0])
        return "UNANSWERABLE"

    answerable_pool["ground_truth"] = answerable_pool["answers"].apply(extract_gt)
    unanswerable_pool["ground_truth"] = "UNANSWERABLE"

    # Sample 800 answerable + 200 unanswerable
    sample_ans = answerable_pool.sample(n=min(800, len(answerable_pool)), random_state=42)
    sample_unans = unanswerable_pool.sample(n=min(200, len(unanswerable_pool)), random_state=42)

    eval_golden_df = pd.concat([sample_ans, sample_unans], ignore_index=True)
    eval_golden_df = eval_golden_df[["id", "title", "context", "question", "ground_truth", "is_answerable"]]
    eval_golden_df = eval_golden_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Created Golden Evaluation Set with {len(eval_golden_df)} queries ({eval_golden_df['is_answerable'].sum()} answerable, {(~eval_golden_df['is_answerable']).sum()} unanswerable).")

    os.makedirs(output_dir, exist_ok=True)
    corpus_out = os.path.join(output_dir, "corpus_chunks.parquet")
    golden_out = os.path.join(output_dir, "eval_golden_set.parquet")

    corpus_df.to_parquet(corpus_out, index=False)
    eval_golden_df.to_parquet(golden_out, index=False)
    print(f"Saved processed corpus to: {corpus_out}")
    print(f"Saved evaluation benchmark to: {golden_out}")

    return corpus_df, eval_golden_df


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_train = os.path.join(base_dir, "data", "raw", "squad_v2_train.parquet")
    raw_val = os.path.join(base_dir, "data", "raw", "squad_v2_val.parquet")
    out_dir = os.path.join(base_dir, "data", "processed")

    build_rag_corpus(raw_train, raw_val, out_dir)
