from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import streamlit as st

from .model import get_client_of_openai

@st.cache_data
@st.cache_resource
def get_rag_chain(file_path):
    chat = get_client_of_openai()

    # 1) 챗봇에 '기억'을 입히기 위한 첫번째 단계
    retriever_of_vectorstore = __get_retriever_of_vectorstore(file_path, chat)

    # 2) 두번째 단계로, 방금 전 생성한 체인을 사용하여 문서를 불러올 수 있는 retriever 체인을 생성합니다.
    rag_chain = __get_rag_chain(chat, retriever_of_vectorstore)
    return rag_chain

def __get_retriever_of_vectorstore(file_path, chat):

    vectorstore = __get_vectorstore(file_path)
    retriever = vectorstore.as_retriever(k=2)

    prompt_of_vectorstore = __get_prompt_of_vectorstore()

    # 이를 토대로 메세지 기록을 기억하는 retriever를 생성합니다.
    history_aware_retriever = create_history_aware_retriever(
        chat, retriever, prompt_of_vectorstore
    )
    return history_aware_retriever

def __get_vectorstore(file_path):
    # load data
    loader = PyPDFLoader(
        file_path
    )
    pages = loader.load_and_split()

    # add data to Chroma
    vectorstore = Chroma.from_documents(pages, OpenAIEmbeddings())
    return vectorstore

def __get_prompt_of_vectorstore():
    contextualize_q_system_prompt = """
        이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 
        이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 
        질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""

    # MessagesPlaceholder: 'chat_history' 입력 키를 사용하여 이전 메세지 기록들을 프롬프트에 포함시킴.
    # 즉 프롬프트, 메세지 기록 (문맥 정보), 사용자의 질문으로 프롬프트가 구성됨. 
    prompt_of_vectorstore = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return prompt_of_vectorstore


def __get_rag_chain(chat, history_aware_retriever):
    qa_system_prompt = """
    질문-답변 업무를 돕는 보조원입니다. 
    질문에 답하기 위해 검색된 내용을 사용하세요. 
    답을 모르면 모른다고 말하세요. 
    답변은 세 문장 이내로 간결하게 유지하세요.

    ## 답변 예시
    📍답변 내용: 
    📍증거: 

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)

    # 결과값은 input, chat_history, context, answer 포함함.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain






