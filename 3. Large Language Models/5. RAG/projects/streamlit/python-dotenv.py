import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
SECRET_ENV = os.getenv("SECRET_ENV")

st.title(f"SECRET_ENV > {SECRET_ENV}")