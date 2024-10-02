import os
import tempfile
from dotenv import load_dotenv
# 환경변수 로딩
load_dotenv()

import streamlit as st

from common.chat import run_chat, init_session_history
from common.model import get_client_of_openai
from common.utils import display_pdf, upload_file
from common.rag import get_retriever

st.title("chatbot.py")

# Initialize chat history
init_session_history(st)

with st.sidebar:

    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            
            with tempfile.TemporaryDirectory() as temp_dir:
                file_key, file_path = upload_file(st, temp_dir, uploaded_file)

                if file_key not in st.session_state.get('file_cache', {}):

                    get_retriever(file_path)

                    from langchain_upstage import ChatUpstage
                    from langchain_core.messages import HumanMessage, SystemMessage

                    chat = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"))

                    # 1) 챗봇에 '기억'을 입히기 위한 첫번째 단계 

                    # 이전의 메시지들과 최신 사용자 질문을 분석해, 문맥에 대한 정보가 없이 혼자서만 봤을때 이해할 수 있도록 질문을 다시 구성함
                    # 즉 새로 들어온 그 질문 자체에만 집중할 수 있도록 다시 재편성
                    from langchain.chains import create_history_aware_retriever
                    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

                    contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 
                    이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 
                    질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""

                    # MessagesPlaceholder: 'chat_history' 입력 키를 사용하여 이전 메세지 기록들을 프롬프트에 포함시킴.
                    # 즉 프롬프트, 메세지 기록 (문맥 정보), 사용자의 질문으로 프롬프트가 구성됨. 
                    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", contextualize_q_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )

                    # 이를 토대로 메세지 기록을 기억하는 retriever를 생성합니다.
                    history_aware_retriever = create_history_aware_retriever(
                        chat, retriever, contextualize_q_prompt
                    )

                    # 2) 두번째 단계로, 방금 전 생성한 체인을 사용하여 문서를 불러올 수 있는 retriever 체인을 생성합니다.
                    from langchain.chains import create_retrieval_chain
                    from langchain.chains.combine_documents import create_stuff_documents_chain

                    qa_system_prompt = """질문-답변 업무를 돕는 보조원입니다. 
                    질문에 답하기 위해 검색된 내용을 사용하세요. 
                    학습한 문서와 관련되지 않은 질문을 물어보면 "죄송합니다. 해당 질문에 대한 답변은 할 수 없습니다."라고 답변해주세요.
                    답을 모르면 억지로 지어내지 말고 답변을 모른다고 말하세요. 
                    답변을 꼭 세 문장 이내로 간결하게 유지하세요.

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

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     
