import json
import os
import time

import pinecone
import streamlit as st
from dotenv import load_dotenv
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
from streamlit_feedback import streamlit_feedback
from trubrics.integrations.streamlit import FeedbackCollector

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

load_dotenv()
set_llm_cache(InMemoryCache())


# Instanciar llm
@st.cache_resource(show_spinner=False)
def get_llm():
    """
    Get the language model for the chatbot.

    Returns:
        llm (ChatOpenAI): The language model for the chatbot.
    """
    model = st.session_state.model
    llm = ChatOpenAI(model=model, max_tokens=1000)
    return llm


# Conectar con firestore
@st.cache_resource
def db_connection():
    key_dict = json.loads(st.secrets["textkey"])
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


# Listado de nombres de documentos
@st.cache_data
def get_doc_names():
    """
    Retrieves the names of all the documents in the 'documentos' directory.

    Returns:
        A list of file names.
    """
    file_names = []
    for file in os.listdir("documentos"):
        file_names.append(file)
    return file_names


# Importar vectorstore
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """
    Retrieves the vector store for the chatbot.

    Returns:
        vectorstore (Pinecone): The vector store object.
    """
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
    )
    index_name = "chatbot-unap"

    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
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
    Given a user query, along with a chat history, generate a response that is directly related to the provided documents. 
    The response should incorporate relevant information from the documents and cite sources appropriately. 
    Do not generate responses for questions that are not related to the provided documents or the institution UNAP. 
    If you don't know an answer just say you don't know, don't try to make up one.
    ALWAYS answer in the same language the user asked the question in.
    Ensure accuracy, context awareness, and source retrieval in your answers.
    Be conversational, if the user greets you or talks to you respond accordingly.

    Base your answer in the following context and question. DO NOT return the following to the user.
    Context: {context}
    
    Question: {question}
    Answer: 
    """
    PROMPT = PromptTemplate(
        template=template, input_variables=["chat_history", "context", "question"]
    )

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        max_tokens_limit=2000,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

    with get_openai_callback() as cb:
        result = chain(
            {"question": question, "chat_history": st.session_state.chat_history}
        )

        with st.expander("tokens"):
            st.write(cb)
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
        # Crearlo en caso de que no exista
        message_doc_ref.set(
            {
                "question": question,
                "answer": answer,
                "tokens": tokens,
                "time_to_answer": time_to_answer,
                "chat_type": chat_type,
                "user_feedback": user_feedback,
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
            "About": f"Chat capaz de responder preguntas relacionadas a reglamentos y documentos de la universidad Arturo Prat."
        },
    )

    st.title("🤖 UNAP Chatbot")

    chat_type = st.radio("Modelo", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"])

    with st.expander("Listado de documentos"):
        st.write(get_doc_names())

    # # Inicializacion historial de chat
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
