import logging
import uuid

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from yaml.loader import SafeLoader

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    # st.set_page_config(initial_sidebar_state="collapsed")
    if "role" not in st.session_state:
        st.session_state.role = "User"
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
    if "authenticator" not in st.session_state:
        with open(".streamlit/auth_config.yaml") as file:
            config = yaml.load(file, Loader=SafeLoader)
        st.session_state.authenticator = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
        )

    if st.session_state.authentication_status:
        st.session_state.role = "Admin"
        with st.sidebar:
            st.write(f"Bienvenido **{st.session_state.name}**")

        st.session_state.authenticator.logout(
            button_name="Cerrar Sesión",
            location="sidebar",
            key=st.session_state.session_id,
        )
    elif st.session_state.authentication_status is False:
        st.session_state.role = "User"
        st.error("Usuario o contraseña incorrectos.")
    else:
        st.session_state.role = "User"
        st.session_state.authenticator.login(
            location="sidebar",
            fields={
                "Form name": "Inicio de sesión administrativo",
                "Username": "Nombre de usuario",
                "Password": "Contraseña",
                "Login": "Iniciar Sesión",
                "Captcha": "Captcha",
            },
        )

    chatbot_page = st.Page(
        page="app_pages/chatbot.py",
        title="Chatbot UNAP",
        icon=":material/smart_toy:",
        default=(st.session_state.role == "User"),
    )
    admin_test_page = st.Page(
        page="app_pages/doc_manager_test.py",
        title="Gestor de Documentos",
        icon=":material/settings:",
        default=(st.session_state.role == "Admin"),
    )

    pages = None
    if st.session_state.role == "User":
        pages = st.navigation([chatbot_page])
    elif st.session_state.role == "Admin":
        pages = st.navigation([chatbot_page, admin_test_page])

    logo = st.logo("logos/logo_wide.png", icon_image="logos/logo_wide.png")
    pages.run()
