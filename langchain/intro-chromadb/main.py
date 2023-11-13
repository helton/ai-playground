from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def build_template():
    system_prompt = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer."""
    begin_instruction, end_instruction = "[INST]", "[/INST]"
    begin_system, end_system = "<>\n", "\n<>\n\n"
    system_prompt = begin_system + system_prompt + end_system
    instruction = """
        {context}

        Question: {question}
        """
    return begin_instruction + system_prompt + instruction + end_instruction


def ingestion():
    loader = PyPDFDirectoryLoader("docs")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    all_docs = text_splitter.split_documents(data)
    persist_directory = "local/data/db/chromadb/index"
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(
        texts=[d.page_content for d in all_docs],
        embedding=embedding,
        persist_directory=persist_directory,
    )
    vectordb.persist()


def search(query: str):
    prompt = PromptTemplate(
        template=build_template(), input_variables=["context", "question"]
    )
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        persist_directory="local/data/db/chromadb/index",
        embedding_function=embeddings,
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    response = qa_chain({"query": query})
    print(response["result"])


if __name__ == "__main__":
    # ingestion()
    search(query="What the document is about?")
