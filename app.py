import streamlit as st

if __name__ == "__main__":
    chatbot_page = st.Page(
        page="pages\chatbot.py",
        title="Chatbot UNAP",
        icon=":material/smart_toy:",
        default=True,
    )
    admin_page = st.Page(
        page="pages\doc_manager.py",
        title="Admin",
        icon=":material/settings:",
        default=False,
    )
    logo = st.logo("logos\\logo_wide.png", icon_image="logos\\logo_wide.png")

    pages = st.navigation([chatbot_page, admin_page], expanded=False)
    pages.run()
