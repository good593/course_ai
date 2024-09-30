import os
from dotenv import load_dotenv
# 환경변수 로딩
load_dotenv()

import streamlit as st

from common.chat import run_chat, init_session_history
from common.model import get_client_of_openai

st.title("chatbot.py")

# Initialize chat history
init_session_history(st)

# 환경변수에 등록된 값 사용
USER_NAME = os.getenv("USER_NAME")

client = get_client_of_openai()

def response_generator(question):
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You must answer in Korean.",
            },
            {
                "role": "user",
                "content": question,  # 사용자의 질문을 입력
            },
        ],
        stream=True,  # 스트림 모드 활성화
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


run_chat(st, USER_NAME+"님 무엇을 도와드릴까요?", response_generator)

