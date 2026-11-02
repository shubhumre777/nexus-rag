
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

TEST_LIMIT = 3
OUTPUT_FILE = "evaluation_results.csv"


async def main():

    # Load the existing FAISS database
    rag = Rag()
    rag.load_vector_db()

    df = pd.read_csv("test_dataset.csv").head(TEST_LIMIT)

    required_columns = {"question", "ground_truth"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV must contain question and ground_truth columns."
        )

    if df.empty:
        raise ValueError("No questions found in the test dataset.")

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")

    # Connect RAGAS to Groq
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    llm = llm_factory(
        "openai/gpt-oss-120b",
        client=client,
    )

    embeddings = embedding_factory(
        "huggingface",
        model="Qwen/Qwen3-Embedding-0.6B",
        interface="modern",
    )

    # Metrics used to evaluate the RAG system
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    retriever = rag.vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    results = []

    for number, (_, row) in enumerate(df.iterrows(), start=1):

        question = str(row["question"])
        reference = str(row["ground_truth"])

        print("\n" + "-" * 60)
        print(f"Question {number}/{len(df)}")
        print(question)

        result = {
            "question": question,
            "ground_truth": reference,
            "answer": None,
            "retrieved_contexts": None,
            "retrieved_sources": None,
        }

        try:
            # Find relevant documents for the question
            docs = retriever.invoke(question)

            contexts = [doc.page_content for doc in docs]

            sources = list(dict.fromkeys(
                str(doc.metadata.get("source", "Unknown"))
                for doc in docs
            ))

            result["retrieved_contexts"] = "\n\n---\n\n".join(contexts)
            result["retrieved_sources"] = ", ".join(sources)

            print("\nRetrieved Contexts:")

            if contexts:
                for i, context in enumerate(contexts, start=1):
                    print(f"\nContext {i}:")
                    print(context)
            else:
                print("No relevant documents retrieved.")

            # Generate an answer using the retrieved context
            answer = rag.ask(
                question,
                "\n\n".join(contexts)
            )

            result["answer"] = answer

            print("\nGenerated Answer:")
            print(answer)

            print("\nExpected Answer:")
            print(reference)

            # Run each evaluation metric
            for metric in metrics:

                name = metric.name

                try:
                    if name == "faithfulness":
                        score = await metric.ascore(
                            user_input=question,
                            response=answer,
                            retrieved_contexts=contexts,
                        )

                    elif name == "answer_relevancy":
                        score = await metric.ascore(
                            user_input=question,
                            response=answer,
                        )

                    elif name in (
                        "context_precision",
                        "context_recall"
                    ):
                        score = await metric.ascore(
                            user_input=question,
                            reference=reference,
                            retrieved_contexts=contexts,
                        )

                    result[name] = score.value

                    print(f"{name}: {score.value:.3f}")

                except Exception as error:
                    result[name] = None
                    print(f"{name} failed: {error}")

        except Exception as error:
            result["error"] = str(error)
            print(f"Question failed: {error}")

        results.append(result)

        # Save results after every question
        results_df = pd.DataFrame(results)

        results_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(f"\nCompleted: {number}/{len(df)}")

    # Calculate the average score for each metric
    if results:

        results_df = pd.DataFrame(results)

        metric_names = [metric.name for metric in metrics]

        print("\n" + "-" * 60)
        print("Average Evaluation Scores:")

        average_scores = results_df[metric_names].mean()

        print(average_scores.round(3))

        print(f"\nResults saved to {OUTPUT_FILE}")

    else:
        print("No evaluation results were generated.")


if __name__ == "__main__":
    asyncio.run(main())
