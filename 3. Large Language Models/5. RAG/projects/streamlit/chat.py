import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOpenAI()

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi, how are you?")
]
response = chat.invoke(messages)
print(response.content)
st.title(f"response > {response.content}")



