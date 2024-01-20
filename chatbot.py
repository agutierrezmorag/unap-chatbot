import json
import time
from datetime import datetime

import pinecone
import pytz
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account
from icecream import ic
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

from documents_manager import get_repo_documents
from utils import config
from utils.callbacks import StreamHandler

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
    llm = ChatOpenAI(
        model=model,
        openai_api_key=config.OPENAI_API_KEY,
        max_tokens=1000,
        streaming=True,
    )
    return llm


# Conectar con firestore
@st.cache_resource
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


# Total de chats
def get_chats_len():
    """
    Returns the number of chats in the 'chats' collection.

    Returns:
        int: The number of chats in the collection.
    """
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


@st.cache_resource(show_spinner=False)
def get_chain():
    """
    Retrieves the conversational retrieval chain for the chatbot.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain object.
    """
    template = """
    Eres un modelo de IA entrenado para proporcionar respuestas precisas y concisas a las consultas de los usuarios.
    Tus respuestas deben basarse en los documentos proporcionados y ser relevantes para la institución Universidad Arturo Prat (UNAP).
    Si la pregunta no es relevante para la UNAP, simplemente indica que no puedes responder a dichas preguntas.

    Si no conoces la respuesta a una pregunta, simplemente indica que no tienes la información.
    Siempre responde en el mismo idioma que la pregunta del usuario.

    Tu objetivo es proporcionar respuestas claras y fáciles de entender.
    Evita los párrafos largos y desglosa la información en oraciones más cortas o puntos si es posible.

    Cuando proporciones información de los documentos explícitamente, recuerda siempre citar la fuente.
    Este es un ejemplo de cómo citar una fuente: "(Reglamento X, Artículo Y, Z)".
    Reemplaza X con el nombre del documento, e Y y Z con el número de los artículos, donde corresponda.
    Si no hay un artículo específico, simplemente indica el nombre del documento.

    Aquí está el historial de chat: {context}
    Aquí está la pregunta del usuario: {question}

    Por favor, genera una respuesta siguiendo estas instrucciones.
    """
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"],
    )

    document_template = """
    Context: {page_content}
    Document name: {file_name}
    """

    DOCUMENT_PROMPT = PromptTemplate(
        template=document_template,
        input_variables=["page_content", "file_name"],
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
        combine_docs_chain_kwargs={
            "prompt": PROMPT,
            "document_prompt": DOCUMENT_PROMPT,
        },
    )

    return chain


# Generacion de respuesta
def answer_question(question, stream_handler):
    """
    Answers a given question using a chatbot model.

    Parameters:
    - question (str): The question to be answered.
    - stream_handler: The stream handler for processing the chatbot response.

    Returns:
    - answer (str): The answer generated by the chatbot.
    - tokens (dict): A dictionary containing information about the tokens used in the chatbot response.
    - source_documents: The source documents used by the chatbot model.
    """
    chain = get_chain()

    with get_openai_callback() as cb:
        result = chain(
            {
                "question": question,
                "chat_history": st.session_state.chat_history,
            },
            callbacks=[stream_handler],
        )

        ic(result)

        tokens = {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost_usd": cb.total_cost,
        }

    st.session_state.chat_history = [(question, result["answer"])]

    answer = result["answer"]
    return answer, tokens, result["source_documents"]


def process_question(prompt, chat_type):
    """
    Process a user's question in the chatbot.

    Args:
        prompt (str): The user's question.
        chat_type (str): The type of chat.

    Returns:
        None
    """
    st.session_state.message_id = str(get_messages_len() + 1)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=logo_path):
        start = time.time()
        stream_handler = StreamHandler(st.empty())
        full_response, tokens, sources = answer_question(
            question=prompt, stream_handler=stream_handler
        )
        end = time.time()

    # Agregar respuesta a historial de chat
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    add_to_db(
        question=prompt,
        answer=full_response,
        tokens=tokens,
        time_to_answer=end - start,
        chat_type=chat_type,
        message_id=st.session_state.message_id,
        sources=sources,
    )


# Funcion generador de id unico
def unique_id_gen(subcollection_ref, base_id):
    """
    Generates a unique ID for a document in a subcollection.

    Args:
        subcollection_ref (SubcollectionReference): Reference to the subcollection.
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
    Add a question and answer to the database.

    Parameters:
    - question (str): The question to be added.
    - answer (str): The answer to the question.
    - tokens (list): List of tokens associated with the question.
    - time_to_answer (float): Time taken to answer the question.
    - chat_type (str): Type of chat.
    - message_id (str): ID of the message.
    - user_feedback (str, optional): Feedback provided by the user. Defaults to None.
    - sources (dic): Documents source names and context.
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
        page_title="Chatbot UNAP 📖",
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

    st.title("🤖 Chatbot UNAP")
    st.caption(
        "Este chatbot puede cometer errores. Si encuentras inexactitudes, reformula tu pregunta o consulta los documentos oficiales."
    )

    docs = get_repo_documents()
    with st.expander("Puedes realizar consultas sobre los siguientes documentos:"):
        for doc in docs:
            st.caption(doc.path.strip("documentos/").strip(".txt"))

    show_pages_from_config()

    chat_type = st.radio("Modelo", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"])

    # Inicializacion historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(get_chats_len() + 1)
    if "message_id" not in st.session_state:
        st.session_state.message_id = str(get_messages_len() + 1)
    if "model" not in st.session_state:
        st.session_state.model = chat_type

    questions = [
        "¿Cuales son las tareas del decano?",
        "¿Que hago en caso de reprobar una asignatura?",
        "Explica en que consiste el trabajo de titulo",
        "¿Cuales son los requisitos para titularse?",
    ]

    qcol1, qcol2 = st.columns(2)
    ex_prompt = ""

    # Mantener historial en caso de rerun de app
    for message in st.session_state.messages:
        if message["role"] == "user":
            avatar = "🧑‍💻"
        else:
            avatar = logo_path

        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Escribe tu pregunta...")

    for question in questions[:2]:
        with qcol1:
            if st.button(question, use_container_width=True):
                ex_prompt = question
    for question in questions[2:]:
        with qcol2:
            if st.button(question, use_container_width=True):
                ex_prompt = question

    if ex_prompt:
        prompt = ex_prompt
    if prompt or ex_prompt:
        process_question(prompt, chat_type)

    if len(st.session_state.messages) > 0:
        streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="Proporciona feedback adicional (Opcional)",
            key=st.session_state.message_id,
            on_submit=update_feedback,
        )


if __name__ == "__main__":
    main()
