from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader


def get_retriever(file_path):
    loader = PyPDFLoader(
        file_path
    )
    pages = loader.load_and_split()

    vectorstore = Chroma.from_documents(pages, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(k=2)
    return retriever
