import asyncio
import os

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rag import Rag

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

load_dotenv()

DATASET_FILE = "test_dataset.csv"
OUTPUT_FILE = "evaluation_results.csv"
EXPECTED_QUESTIONS = 50
TOP_K = 5


def is_answerable(value):
    """Read True/False values from the CSV consistently."""
    return str(value).strip().lower() in {"true", "1", "yes"}


async def main():

    # Load the BGE-based FAISS index and the same hybrid retriever used by the app.
    rag = Rag()
    rag.load_vector_db()

    df = pd.read_csv(DATASET_FILE)

    required_columns = {"question", "ground_truth", "answerable"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{DATASET_FILE} must contain: {', '.join(sorted(required_columns))}"
        )

    if len(df) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} questions, but found {len(df)}."
        )

    if df["question"].isna().any() or df["ground_truth"].isna().any():
        raise ValueError("Dataset contains an empty question or ground_truth.")

    answerable_count = sum(is_answerable(value) for value in df["answerable"])

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment or .env file.")

    # Groq is used as the RAGAS evaluation judge.
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    llm = llm_factory(
        "openai/gpt-oss-120b",
        client=client,
    )

    # Use the same BGE family as the application for Answer Relevancy scoring.
    evaluator_embeddings = embedding_factory(
        "huggingface",
        model="BAAI/bge-small-en-v1.5",
        interface="modern",
    )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    # Results are saved after each question so completed work is not lost.
    results = []

    print(f"Dataset: {len(df)} questions")
    print(f"Answerable: {answerable_count} | Unanswerable: {len(df) - answerable_count}")
    print(f"Retriever: FAISS + BM25 | Top-K: {TOP_K}")

    for number, (_, row) in enumerate(df.iterrows(), start=1):

        question = str(row["question"])
        reference = str(row["ground_truth"])
        answerable = is_answerable(row["answerable"])

        print("\n" + "-" * 60)
        print(f"Question {number}/{len(df)}")
        print(question)

        result = {
            "question_number": number,
            "question": question,
            "ground_truth": reference,
            "answerable": answerable,
            "category": row.get("category", ""),
            "source_file": row.get("source_file", ""),
            "answer": None,
            "retrieved_contexts": None,
            "retrieved_sources": None,
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "error": None,
        }

        try:
            # Use the app's hybrid retriever (not FAISS-only retrieval).
            docs = rag.ensemble_retriever.invoke(question)
            docs = docs[:TOP_K]

            contexts = [doc.page_content for doc in docs]
            sources = list(dict.fromkeys(
                str(doc.metadata.get("source", "Unknown"))
                for doc in docs
            ))

            result["retrieved_contexts"] = "\n\n---\n\n".join(contexts)
            result["retrieved_sources"] = ", ".join(sources)

            # Generate an answer using the same RAG prompt and LLM chain as the app.
            answer = rag.ask(question, "\n\n".join(contexts))
            result["answer"] = answer

            for metric in metrics:

                metric_name = metric.name

                # Context Precision / Recall need answer references, so score only
                # questions whose answers are present in the source documents.
                if metric_name in ("context_precision", "context_recall") and not answerable:
                    continue

                try:
                    if metric_name == "faithfulness":
                        score = await metric.ascore(
                            user_input=question,
                            response=answer,
                            retrieved_contexts=contexts,
                        )

                    elif metric_name == "answer_relevancy":
                        score = await metric.ascore(
                            user_input=question,
                            response=answer,
                        )

                    else:
                        score = await metric.ascore(
                            user_input=question,
                            reference=reference,
                            retrieved_contexts=contexts,
                        )

                    result[metric_name] = score.value
                    print(f"{metric_name}: {score.value:.3f}")

                except Exception as error:
                    print(f"{metric_name} failed: {error}")

        except Exception as error:
            result["error"] = str(error)
            print(f"Question failed: {error}")

        results.append(result)

        # Save progress after every question.
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {number}/{len(df)} questions to {OUTPUT_FILE}")

    # Print the average of successful scores; missing/failed scores are ignored.
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("50-QUESTION RAGAS EVALUATION SUMMARY")
    print("=" * 60)

    metric_targets = {
        "faithfulness": len(df),
        "answer_relevancy": len(df),
        "context_precision": answerable_count,
        "context_recall": answerable_count,
    }

    for metric_name, target_count in metric_targets.items():
        scores = pd.to_numeric(
            results_df[metric_name],
            errors="coerce"
        ).dropna()

        average = scores.mean() if not scores.empty else float("nan")

        print(
            f"{metric_name}: {average:.3f} "
            f"({len(scores)}/{target_count} scored)"
            if not pd.isna(average)
            else f"{metric_name}: no scores ({len(scores)}/{target_count} scored)"
        )

    print(f"\nQuestion-level results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
