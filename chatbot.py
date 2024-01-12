import json
import os
import time
from datetime import datetime

import pinecone
import pytz
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account
from langchain.cache import InMemoryCache
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from st_pages import show_pages_from_config
from streamlit_feedback import streamlit_feedback

from utils import config

set_llm_cache(InMemoryCache())

logo_path = "logos/unap_negativo.png"


# Instanciar llm
@st.cache_resource(show_spinner=False)
def get_llm():
    """
    Get the language model for the chatbot.

    Returns:
        llm (ChatOpenAI): The language model for the chatbot.
    """
    model = st.session_state.model
    llm = ChatOpenAI(model=model, openai_api_key=config.OPENAI_API_KEY, max_tokens=1000)
    return llm


# Conectar con firestore
@st.cache_resource
def db_connection():
    key_dict = json.loads(config.FIRESTORE_TEXT_KEY)
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db


# Total de chats
def get_chats_len():
    chats_ref = db_connection().collection("chats").get()
    return len(chats_ref)


def get_messages_len():
    """
    Get the length of the messages in a chat.

    Returns:
        int: The number of messages in the chat.
    """
    chat_id = st.session_state.chat_id
    message_ref = (
        db_connection()
        .collection("chats")
        .document(chat_id)
        .collection("messages")
        .get()
    )
    return len(message_ref)


# Importar vectorstore
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """
    Retrieves the vector store for the chatbot.

    Returns:
        vectorstore (Pinecone): The vector store object.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    pinecone.init(
        api_key=config.PINECONE_API_KEY,
        environment=config.PINECONE_ENV,
    )

    vectorstore = Pinecone.from_existing_index(config.PINECONE_INDEX_NAME, embeddings)
    return vectorstore


# Generacion de respuesta
def answer_question(question):
    """
    Generate a response to a user's question based on a given chat history and context.

    Args:
        question (str): The user's question.

    Returns:
        tuple: A tuple containing the generated answer and token information.
    """

    template = """
    You are an AI model trained to provide accurate and concise answers to user queries. \
    Your responses should be based on the provided documents and relevant to the institution UNAP. 
    If the question is not relevant to UNAP, simply state that you are not able to answer such questions. 

    If the answer to a question is not found in the documents, simply state that you don't have the information. \
    Always respond in the same language as the user's question. 

    Your goal is to provide clear, easy-to-understand answers. \
    Avoid long paragraphs and break down information into shorter sentences or bullet points if possible. 

    When you provide information from the documents, always cite the source. 

    Here is the chat history: {context}
    Here is the user's question: {question}
    And here are the documents you should use: {sources}

    Please generate a response following these instructions.
    """
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question", "sources"],
    )

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        max_tokens_limit=2000,
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

    docs = retriever.get_relevant_documents(question, search_kwargs={"k": 5})

    source_doc_names = set()
    for document in docs:
        file_path = document.metadata["source"]
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        formatted_name = " ".join(word.capitalize() for word in file_name.split("_"))

        # Check if the name has already been printed
        if formatted_name not in source_doc_names:
            print(formatted_name)
            source_doc_names.add(formatted_name)

    if len(source_doc_names) == 1:
        source_doc_names_str = next(iter(source_doc_names))
    else:
        source_doc_names_str = ", ".join(source_doc_names)

    with get_openai_callback() as cb:
        result = chain(
            {
                "question": question,
                "chat_history": st.session_state.chat_history,
                "sources": source_doc_names_str,
            }
        )

        print(cb)

        tokens = {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost_usd": cb.total_cost,
        }

    st.session_state.chat_history = [(question, result["answer"])]

    answer = result["answer"]
    return answer, tokens


# Registrar datos en la base de datos
def add_to_db(
    question, answer, tokens, time_to_answer, chat_type, message_id, user_feedback=None
):
    """
    Add a question and answer to the database.

    Parameters:
    - question (str): The question to be added.
    - answer (str): The answer to the question.
    - tokens (list): List of tokens associated with the question.
    - time_to_answer (float): Time taken to answer the question.
    - chat_type (str): Type of chat.
    - message_id (str): ID of the message.
    - user_feedback (str, optional): Feedback provided by the user. Defaults to None.
    """

    chat_id = st.session_state.chat_id
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


def update_feedback(feedback):
    """
    Updates the user feedback for a specific message in the chat.

    Parameters:
    feedback (str): The user feedback to be updated.

    Returns:
    None
    """

    chat_id = st.session_state.chat_id
    message_id = st.session_state.message_id
    db = db_connection()

    chats_ref = db.collection("chats")
    chat_doc_ref = chats_ref.document(chat_id)

    message_ref = chat_doc_ref.collection("messages")
    message_doc_ref = message_ref.document(message_id)

    message_doc_ref.update({"user_feedback": feedback})


def main():
    st.set_page_config(
        page_title="UNAP Chatbot 📖",
        page_icon="🤖",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": "Chat capaz de responder preguntas relacionadas a reglamentos y documentos de la universidad Arturo Prat."
        },
    )

    st.markdown(
        """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-top: auto;
            width: 100%;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image(logo_path)

    st.title("🤖 UNAP Chatbot")
    st.caption(
        "Este chatbot puede cometer errores. Si encuentras inexactitudes, reformula tu pregunta o consulta los documentos oficiales."
    )

    show_pages_from_config()

    chat_type = st.radio("Modelo", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"])

    # Inicializacion historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! ¿Como te puedo ayudar?"}
        ]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(get_chats_len() + 1)
    if "message_id" not in st.session_state:
        st.session_state.message_id = str(get_messages_len() + 1)
    if "model" not in st.session_state:
        st.session_state.model = chat_type

    # Mantener historial en caso de rerun de app
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Escribe tu pregunta...")

    if prompt:
        st.session_state.message_id = str(get_messages_len() + 1)
        # Agregar input de usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Mostrar input en su contenedor
        with st.chat_message("user"):
            st.markdown(prompt)
        # Mostrar respuesta del LLM
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            start = time.time()
            with st.spinner("Generando respuesta..."):
                full_response, tokens = answer_question(question=prompt)
            message_placeholder.markdown(full_response)
            end = time.time()

        # Agregar respuesta del LLM al historial
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        add_to_db(
            question=prompt,
            answer=full_response,
            tokens=tokens,
            time_to_answer=end - start,
            chat_type=chat_type,
            message_id=st.session_state.message_id,
        )

    # Pasada la primera respuesta NO entra a la funcion
    streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="Proporciona feedback adicional (Opcional)",
        key=st.session_state.message_id,
        on_submit=update_feedback,
    )


if __name__ == "__main__":
    main()
