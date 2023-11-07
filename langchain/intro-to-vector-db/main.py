import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.llms.openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENVIRONMENT'])

if __name__ == '__main__':
    loader = TextLoader('data/blog.txt', encoding='utf8')
    document = loader.load()
    print(document)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    doc_search = Pinecone.from_documents(texts, embeddings, index_name=os.environ['PINECONE_INDEX_NAME'])

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=doc_search.as_retriever(),
        return_source_documents=True
    )
    query = "What is a Vector DB? Give me a 15 word answer for a beginner."
    result = qa({"query": query})
    print(result)
