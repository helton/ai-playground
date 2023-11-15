import os
import shutil

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()


def get_full_model_path(path: str):
    return rf'{os.environ["BASE_MODEL_DIR"]}\{path}'


models = {
    "mistral-7b-instruct": {
        "type": "mistral",
        "path": get_full_model_path(
            "TheBloke\Mistral-7B-Instruct-v0.1-GGUF\mistral-7b-instruct-v0.1.Q5_K_M.gguf"
        ),
    },
    "llama-2-7b-chat": {
        "type": "llama",
        "path": get_full_model_path(
            "TheBloke\Llama-2-7B-Chat-GGML\llama-2-7b-chat.ggmlv3.q5_K_M.bin"
        ),
    },
    "zephyr-7b-beta": {
        "type": "mistral",
        "path": get_full_model_path(
            "TheBloke\zephyr-7B-beta-GGUF\zephyr-7b-beta.Q5_K_M.gguf"
        ),
    },
}

base_config = {
    "max_new_tokens": 2048,
    "context_length": 4096,
    "repetition_penalty": 1.1,
}
faiss_index = os.environ["FAISS_INDEX"]


def get_llm(model):
    llm = CTransformers(
        model_type=model["type"], model=model["path"], config=base_config
    )
    return llm


def get_embeddings():
    return GPT4AllEmbeddings()


def get_prompt_template():
    system_prompt = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer."""
    begin_instruction, end_instruction = "[INST]", "[/INST]"
    begin_system, end_system = "<>\n", "\n<>\n\n"
    system_prompt = begin_system + system_prompt + end_system
    instruction = """
        {context}

        Question: {question}
        """
    template = begin_instruction + system_prompt + instruction + end_instruction
    return PromptTemplate(template=template, input_variables=["context", "question"])


def store_docs():
    shutil.rmtree(faiss_index)
    doc_paths = ["data/docs/react.pdf", "data/docs/chain-of-thought.pdf"]
    for pdf_path in doc_paths:
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(documents)
        embeddings = get_embeddings()

        if os.path.exists(faiss_index) and os.path.isdir(faiss_index):
            db = FAISS.load_local(folder_path=faiss_index, embeddings=embeddings)
            db.add_documents(documents=docs, embedding=embeddings)
        else:
            db = FAISS.from_documents(documents=docs, embedding=embeddings)
        db.save_local(faiss_index)
    print(f"{len(doc_paths)} documents stored in {faiss_index}")


def query_llm(query: str, model_name: str):
    llm = get_llm(model=models[model_name])
    return llm(query)


def chain(query: str, model_name: str, chain_type: str = "stuff"):
    embeddings = get_embeddings()
    llm = get_llm(model=models[model_name])
    prompt = get_prompt_template()

    db = FAISS.load_local(faiss_index, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    response = qa_chain({"query": query})
    return {"answer": response["result"].strip(), "documents": response["source_documents"]}


if __name__ == "__main__":
    store_docs()
    # print(query_llm(query="AI is going to", model_name="mistral-7b-instruct"))
    # print(query_llm(query="AI is going to", model_name="llama-2-7b-chat"))
    # print(query_llm(query="AI is going to", model_name="zephyr-7b-beta"))
    result = chain(
        query="How CoT compares to ReAct?",
        model_name="llama-2-7b-chat",
        chain_type="stuff",
    )
    print(result["answer"])
    # print(result["documents"])
