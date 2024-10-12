import streamlit as st

from app_pages.chatbot import chatbot_page
from app_pages.doc_manager import doc_manager_page

if __name__ == "__main__":
    chatbot_page = st.Page(
        page=chatbot_page,
        title="Chatbot UNAP",
        icon=":material/smart_toy:",
        default=True,
    )
    admin_page = st.Page(
        page=doc_manager_page,
        title="Admin",
        icon=":material/settings:",
        default=False,
    )
    logo = st.logo("logos\\logo_wide.png", icon_image="logos\\logo_wide.png")

    pages = st.navigation([chatbot_page, admin_page], expanded=False)
    pages.run()
