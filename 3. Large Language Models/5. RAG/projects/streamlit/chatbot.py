import os
from dotenv import load_dotenv
# 환경변수 로딩
load_dotenv()

import random
import time
import streamlit as st

from common.chat import run_chat, init_session_history

st.title("chatbot.py")

# Initialize chat history
init_session_history(st)

# 환경변수에 등록된 값 사용
USER_NAME = os.getenv("USER_NAME")

# Streamed response emulator
def response_generator(prompt):
    response = random.choice(
        [
            "무엇을 도와드릴까요?",
            "주말인데, 여행지 추천해드릴까요?",
            "이렇게 좋은 날에는 공원산책 어떠신가요?",
        ]
    )
    responses = [USER_NAME+"님, "] + response.split() 
    for word in responses:
        yield word + " "
        time.sleep(0.05)

run_chat(st, USER_NAME+"님 무엇을 도와드릴까요?", response_generator)

