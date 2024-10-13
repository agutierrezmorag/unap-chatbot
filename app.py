import uuid

import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

if __name__ == "__main__":
    if "role" not in st.session_state:
        st.session_state.role = "user"
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "run_id" not in st.session_state:
        st.session_state.run_id = ""
    if "msgs" not in st.session_state:
        st.session_state.msgs = StreamlitChatMessageHistory(key="msgs")
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            chat_memory=st.session_state.msgs,
            return_messages=True,
        )
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    chatbot_page = st.Page(
        page="app_pages/chatbot.py",
        title="Chatbot UNAP",
        icon=":material/smart_toy:",
        default=True,
    )
    admin_page = st.Page(
        page="app_pages/doc_manager.py",
        title="Admin",
        icon=":material/settings:",
        default=False,
    )
    logo = st.logo("logos\\logo_wide.png", icon_image="logos\\logo_wide.png")

    pages = st.navigation([chatbot_page, admin_page])
    pages.run()
