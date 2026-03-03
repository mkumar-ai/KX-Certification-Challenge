# evaluate.py
# Purpose: RAGAS evaluation — Baseline vs HyDE retriever comparison

import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from agent import run_agent
from retriever import retrieve
from retriever_hyde import retrieve_hyde

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOLDEN_DATASET = "data/golden_dataset.csv"

# ── RAGAS wrappers ────────────────────────────────────────────────────────────
ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
)

METRICS = [faithfulness, context_precision, context_recall, answer_relevancy]
METRIC_NAMES = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]


def build_ragas_dataset(df: pd.DataFrame, retriever_fn, label: str) -> Dataset:
    """Run all questions through the agent + retriever and build RAGAS dataset."""
    questions, ground_truths, answers, contexts = [], [], [], []

    for i, row in df.iterrows():
        question     = row["question"]
        ground_truth = row["ground_truth_answer"]

        print(f"  [{label}] [{i+1}/{len(df)}] {question[:80]}...")

        answer        = run_agent(question)
        retrieved     = retriever_fn(question, top_k=4)
        context_texts = [r["text"] for r in retrieved]

        questions.append(question)
        ground_truths.append(ground_truth)
        answers.append(answer)
        contexts.append(context_texts)

    return Dataset.from_dict({
        "user_input":         questions,
        "response":           answers,
        "retrieved_contexts": contexts,
        "reference":          ground_truths,
    })


def run_ragas(dataset: Dataset) -> pd.DataFrame:
    results = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return results.to_pandas()


def run_evaluation():
    print("Loading golden dataset...")
    df = pd.read_csv(GOLDEN_DATASET)
    print(f"  -> {len(df)} questions loaded\n")

    # ── Baseline evaluation ───────────────────────────────────────────────────
    print("=" * 60)
    print("RUNNING BASELINE EVALUATION...")
    print("=" * 60)
    baseline_dataset = build_ragas_dataset(df, retrieve, label="Baseline")
    baseline_df      = run_ragas(baseline_dataset)
    baseline_df.to_csv("data/ragas_baseline_results.csv", index=False)
    print("Baseline results saved to data/ragas_baseline_results.csv")

    # ── HyDE evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RUNNING HyDE EVALUATION...")
    print("=" * 60)
    hyde_dataset = build_ragas_dataset(df, retrieve_hyde, label="HyDE")
    hyde_df      = run_ragas(hyde_dataset)
    hyde_df.to_csv("data/ragas_hyde_results.csv", index=False)
    print("HyDE results saved to data/ragas_hyde_results.csv")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON: BASELINE vs HyDE")
    print("=" * 60)

    available_metrics = [m for m in METRIC_NAMES if m in baseline_df.columns and m in hyde_df.columns]

    comparison = pd.DataFrame({
        "Metric":   available_metrics,
        "Baseline": [baseline_df[m].mean() for m in available_metrics],
        "HyDE":     [hyde_df[m].mean()     for m in available_metrics],
    })
    comparison["Delta"] = comparison["HyDE"] - comparison["Baseline"]
    comparison["Change"] = comparison["Delta"].apply(
        lambda x: f"▲ +{x:.3f}" if x > 0 else f"▼ {x:.3f}"
    )

    pd.set_option("display.float_format", "{:.3f}".format)
    print(comparison[["Metric", "Baseline", "HyDE", "Change"]].to_string(index=False))
    print("=" * 60)

    comparison.to_csv("data/ragas_comparison.csv", index=False)
    print("Comparison saved to data/ragas_comparison.csv")


if __name__ == "__main__":
    run_evaluation()
