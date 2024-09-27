
from .constant import CHATBOT_ROLE
from .chat_history import add_content_in_history

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



