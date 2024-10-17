import locale
import logging
import uuid

import streamlit as st
import yaml
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from yaml.loader import SafeLoader

from utils.custom_authenticator import CustomAuthenticate

logging.basicConfig(level=logging.INFO)
locale.setlocale(locale.LC_TIME, "es_CL.UTF-8")

if __name__ == "__main__":
    st.set_page_config(initial_sidebar_state="collapsed")
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
    if "suggested_question" not in st.session_state:
        st.session_state.suggested_question = None
    if "authenticator" not in st.session_state:
        with open(".streamlit/auth_config.yaml") as file:
            st.session_state.auth_config = yaml.load(file, Loader=SafeLoader)
        st.session_state.authenticator = CustomAuthenticate(
            st.session_state.auth_config["credentials"],
            st.session_state.auth_config["cookie"]["name"],
            st.session_state.auth_config["cookie"]["key"],
            st.session_state.auth_config["cookie"]["expiry_days"],
        )
    if "logout_key" not in st.session_state:
        st.session_state.logout_key = str(uuid.uuid4())
    if "reset_password_key" not in st.session_state:
        st.session_state.reset_password_key = str(uuid.uuid4())

    if st.session_state.authentication_status:
        st.session_state.role = "Admin"
        with st.sidebar:
            st.write(f"Bienvenido **{st.session_state.name}**")
        st.session_state.authenticator.logout(
            button_name="Cerrar Sesión",
            location="sidebar",
            key=st.session_state.logout_key,
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
        default=True,
    )
    docs_manager_page = st.Page(
        page="app_pages/doc_manager.py",
        title="Gestor de Documentos",
        icon=":material/settings:",
    )
    change_password_page = st.Page(
        page="app_pages/change_password.py",
        title="Cambiar Contraseña",
        icon=":material/lock:",
    )
    create_account_page = st.Page(
        page="app_pages/create_account.py",
        title="Crear Cuenta",
        icon=":material/person_add:",
    )

    user_pages = {"Chatbot": [chatbot_page]}
    admin_pages = {
        "Documentos": [docs_manager_page],
        "Cuenta": [change_password_page, create_account_page],
    }

    pages = None
    if st.session_state.role == "User":
        pages = st.navigation(user_pages)
    elif st.session_state.role == "Admin":
        pages = st.navigation(
            user_pages | admin_pages,
        )

    logo = st.logo(
        "logos/logo_wide.png",
        icon_image="logos/logo_small.png",
        size="large",
        link="https://www.unap.cl/",
    )
    pages.run()
