from datetime import datetime
import json

import pytz
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account

from utils import config


# Conectar con firestore
@st.cache_resource
def db_connection():
    """
    Establishes a connection to the Firestore database.

    Returns:
        db (google.cloud.firestore.Client): The Firestore client object.
    """
    key_dict = json.loads(config.FIRESTORE_TEXT_KEY)
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db


# Total de chats
def get_chats_len():
    """
    Get the number of chats in the Firestore database.

    Returns:
        int: The number of chats.
    """
    chats_ref = db_connection().collection("chats").get()
    return len(chats_ref)


def get_messages_len(chat_id):
    """
    Get the number of messages in a chat.

    Args:
        chat_id (str): The ID of the chat.

    Returns:
        int: The number of messages in the chat.
    """
    message_ref = (
        db_connection()
        .collection("chats")
        .document(chat_id)
        .collection("messages")
        .get()
    )
    return len(message_ref)


# Funcion generador de id unico
def unique_id_gen(subcollection_ref, base_id):
    """
    Generate a unique ID for a document in a Firestore subcollection.

    Args:
        subcollection_ref (firestore.CollectionReference): Reference to the subcollection.
        base_id (str): Base ID for the document.

    Returns:
        str: Unique ID for the document.

    """
    i = 1
    while True:
        document_id = f"{base_id}_{i}"
        document_ref = subcollection_ref.document(document_id)
        if not document_ref.get().exists:
            return document_id
        i += 1


# Registrar datos en la base de datos
def add_to_db(
    chat_id,
    question,
    answer,
    tokens,
    time_to_answer,
    chat_type,
    message_id,
    sources,
    user_feedback=None,
):
    """
    Adds a chat message to the Firestore database.

    Parameters:
    - chat_id (str): The ID of the chat.
    - question (str): The question asked in the chat message.
    - answer (str): The answer to the question.
    - tokens (list): The tokens extracted from the question.
    - time_to_answer (float): The time taken to answer the question.
    - chat_type (str): The type of chat.
    - message_id (str): The ID of the chat message.
    - sources (list): The sources associated with the chat message.
    - user_feedback (str, optional): The feedback provided by the user. Defaults to None.
    """
    db = db_connection()

    chats_ref = db.collection("chats")
    chat_doc_ref = chats_ref.document(chat_id)

    # Revisar si documento con chat_id existe
    chat_doc = chat_doc_ref.get()
    if not chat_doc.exists:
        # Crearlo en caso de que no exista
        chat_doc_ref.set({})

    # Agregar pregunta y respuesta a sub coleccion messages
    messages_ref = chat_doc_ref.collection("messages")
    message_doc_ref = messages_ref.document(message_id)

    # Revisar si documento con chat_id existe
    message_doc = message_doc_ref.get()
    if not message_doc.exists:
        local_tz = pytz.timezone("America/Santiago")
        local_time = datetime.now(local_tz)
        # Crearlo en caso de que no exista
        message_doc_ref.set(
            {
                "question": question,
                "answer": answer,
                "tokens": tokens,
                "time_to_answer": time_to_answer,
                "chat_type": chat_type,
                "user_feedback": user_feedback,
                "submission_time": local_time,
            }
        )

    # Crea un documento por contexto, en caso de haber 2 o mas contextos provinientes del
    # mismo documento los almacena en documentos diferentes
    sources_ref = message_doc_ref.collection("sources")

    for document in sources:
        source_name = document.metadata["file_name"]

        unique_id = unique_id_gen(sources_ref, source_name[:-4])

        source_doc_ref = sources_ref.document(unique_id)
        source_docs = source_doc_ref.get()

        if not source_docs.exists:
            source_doc_ref.set(
                {
                    "source_name": source_name[:-4],
                    "context": document.page_content,
                }
            )


def update_feedback(feedback, chat_id, message_id):
    """
    Updates the user feedback for a specific message in a chat.

    Args:
        feedback (str): The user feedback to be updated.
        chat_id (str): The ID of the chat.
        message_id (str): The ID of the message.

    Returns:
        None
    """
    db = db_connection()

    chats_ref = db.collection("chats")
    chat_doc_ref = chats_ref.document(chat_id)

    message_ref = chat_doc_ref.collection("messages")
    message_doc_ref = message_ref.document(message_id)

    message_doc_ref.update({"user_feedback": feedback})
