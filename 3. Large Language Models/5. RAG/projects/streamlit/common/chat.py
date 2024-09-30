import uuid

from .constant import CHATBOT_ROLE

def run_chat(streamlit, default_message, assistant=None):
    # React to user input
    if prompt := streamlit.chat_input(default_message):
        # Display user message in chat message container
        with streamlit.chat_message(CHATBOT_ROLE.user.name):
            streamlit.markdown(prompt)
        # Add user message to chat history
        add_content_in_history(streamlit, CHATBOT_ROLE.user.name, prompt) 

        if assistant:
            # Display assistant response in chat message container
            with streamlit.chat_message(CHATBOT_ROLE.assistant.name):
                response = streamlit.write_stream(assistant(prompt))

            # Add assistant response to chat history
            add_content_in_history(streamlit, CHATBOT_ROLE.assistant.name, response)

# Initialize chat history
def init_session_history(streamlit):
    if "id" not in streamlit.session_state:
        streamlit.session_state.id = uuid.uuid4()
        streamlit.session_state.file_cache = {}

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

