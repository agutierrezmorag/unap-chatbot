import json

import pinecone
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account
from st_pages import show_pages_from_config

from utils import config


def db_connection():
    """
    Establishes a connection to the Firestore database.

    Returns:
        db (google.cloud.firestore.Client): The Firestore database client.
    """
    key_dict = json.loads(config.FIRESTORE_TEXT_KEY)
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db


def check_firestore():
    """
    Checks if the Firestore database is available.

    Returns:
        str: 'success' if the database is available, 'warning' if the connection is successful but retrieving a document fails, 'failure' if the connection fails.
    """
    try:
        db = db_connection()
        try:
            doc = db.collection("chats").limit(1).stream()
            return "success"
        except Exception as e:
            print(e)
            return "warning"
    except Exception as e:
        print(e)
        return "failure"


def check_pinecone():
    """
    Checks if the Pinecone index exists.

    Returns:
        bool: True if the index exists, False otherwise.
    """
    pinecone.init(api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENV)

    try:
        index_info = pinecone.describe_index(config.PINECONE_INDEX_NAME)
        return "success"
    except Exception as e:
        print(e)
        return "failure"


def main():
    st.set_page_config(
        page_title="Disponibilidad de servicios",
        page_icon="💻",
        initial_sidebar_state="collapsed",
    )

    st.title("💻 Disponibilidad de servicios")
    st.markdown(
        "Revisa el estado de cada uno de los servicios necesarios para el correcto funcionamiento de la aplicación."
    )

    show_pages_from_config()

    availability_data = {"Pinecone": check_pinecone(), "Firestore": check_firestore()}

    for service, status in availability_data.items():
        if status == "success":
            st.success(f"{service} está disponible", icon="✅")
        elif status == "warning":
            st.warning(f"{service} presenta problemas", icon="⚠️")
        else:
            st.error(f"{service} no está disponible", icon="❌")


if __name__ == "__main__":
    main()
