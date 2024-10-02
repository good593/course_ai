import streamlit as st
from openai import OpenAI

@st.cache_resource
def get_client_of_openai():
    return OpenAI()

