import os
from llama_index import SimpleWebPageReader, VectorStoreIndex


def main(url: str) -> None:
    documents = SimpleWebPageReader(html_to_text=True).load_data([url])
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is LlamaIndex?")
    print(response)


if __name__ == "__main__":
    main(
        url="https://cbarkinozer.medium.com/an-overview-of-the-llamaindex-framework-9ee9db787d16"
    )
