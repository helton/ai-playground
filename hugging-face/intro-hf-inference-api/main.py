import math
import os
from typing import List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from tabulate import tabulate

load_dotenv()

client = InferenceClient(token=os.environ["HF_TOKEN"])

# type alias
Embedding = List[float]


def embed(text: str, model: str) -> Embedding:
    return list(client.feature_extraction(model=model, text=text))


def guard_vector_dimension(e1, e2):
    if len(e1) != len(e2):
        raise ValueError(
            f"Vector embeddings must have the same dimensions: {len(e1)} vs {len(e2)}"
        )


def dot_product(e1, e2):
    guard_vector_dimension(e1, e2)
    return sum(x * y for x, y in zip(e1, e2))


def cosine_distance(e1, e2):
    guard_vector_dimension(e1, e2)
    dot_prod = dot_product(e1, e2)
    norm_e1 = math.sqrt(sum(x**2 for x in e1))
    norm_e2 = math.sqrt(sum(x**2 for x in e2))
    return 1 - dot_prod / (norm_e1 * norm_e2) if norm_e1 != 0 and norm_e2 != 0 else 0.0


def cosine_similarity(e1, e2):
    return 1 - cosine_distance(e1, e2)


def euclidean_distance(e1, e2):
    guard_vector_dimension(e1, e2)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(e1, e2)))


def euclidean_similarity(e1, e2):
    return 1 / (1 + euclidean_distance(e1, e2))


def compare_embeddings(models: List[str], text1: str, text2: str, digits: int = 7):
    headers = [
        "Model",
        "Dot Product",
        "Cosine Distance",
        "Cosine Similarity",
        "Euclidean Distance",
        "Euclidean Similarity",
    ]
    result = []
    for model in models:
        e1 = embed(model=model, text=text1)
        e2 = embed(model=model, text=text2)
        row = [
            model,
            round(dot_product(e1, e2), digits),
            round(cosine_distance(e1, e2), digits),
            round(cosine_similarity(e1, e2), digits),
            round(euclidean_distance(e1, e2), digits),
            round(euclidean_similarity(e1, e2), digits),
        ]
        result.append(row)

    return headers, result


def run(text1: str, text2: str):
    print(":::Comparing:::")
    print(f"{text1=}")
    print(f"{text2=}")
    print()
    headers, result = compare_embeddings(
        models=["sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-small-v2"],
        text1=text1,
        text2=text2,
        digits=8,
    )
    table = tabulate(result, headers=headers, tablefmt="simple")
    print(table)
    print()
    print("=" * 150)
    print()


if __name__ == "__main__":
    run(text1="This is a happy person", text2="This is a happy person")
    run(text1="This is a happy person", text2="This is a sad person")
    run(text1="An apple fell from the tree", text2="The sky is blue")
