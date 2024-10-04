import os
import tempfile
from dotenv import load_dotenv
# 환경변수 로딩
load_dotenv()

import streamlit as st

from common.chat import init_session_history
from common.utils import display_pdf, upload_file
from common.rag import get_rag_chain

st.title("chatbot.py")

# Initialize chat history
init_session_history(st)

with st.sidebar:

    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # upload file
                file_key, file_path = upload_file(st, temp_dir, uploaded_file)
                # create rag chain
                rag_chain = get_rag_chain(file_path)
                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     
