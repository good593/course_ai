
from .constant import CHATBOT_ROLE

# Initialize chat history
def init_session_history(streamlit):
    if "messages" not in streamlit.session_state:
        streamlit.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in streamlit.session_state.messages:
        if message["role"] in CHATBOT_ROLE.__members__:
            with streamlit.chat_message(message["role"]):
                streamlit.markdown(message["content"])

# Add assistant response to chat history
def add_content_in_history(streamlit, role, content):
    if role in CHATBOT_ROLE.__members__:
        streamlit.session_state.messages.append({"role": role, "content": content}) 