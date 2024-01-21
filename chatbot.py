import time

import pinecone
import streamlit as st
from icecream import ic
from langchain.cache import InMemoryCache
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.globals import set_llm_cache
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from st_pages import show_pages_from_config
from streamlit_feedback import streamlit_feedback

from documents_manager import get_repo_documents
from utils import config
from utils.callbacks import (
    PrintRetrievalHandler,
    StreamHandler,
    TokenUsageTrackingCallbackHandler,
)
from utils.firestore import add_to_db, get_chats_len, get_messages_len, update_feedback

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
        callbacks=[TokenUsageTrackingCallbackHandler(model_name=model)],
    )
    return llm


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
    Si ya has citado una fuente, no es necesario volver a citarla. Puedes simplemente indicar que la información proviene de la misma fuente. 

    Formula tu respuesta a partir de los siguientes documentos: {context}
    Aquí está la pregunta del usuario: {question}

    Por favor, genera una respuesta siguiendo estas instrucciones.
    """
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"],
    )

    document_template = """
    Contenido del documento: {page_content}
    Nombre del documento, formatealo adecuadamente y usalo cuando cites: {file_name}
    """

    DOCUMENT_PROMPT = PromptTemplate(
        template=document_template,
        input_variables=["page_content", "file_name"],
    )

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    memory = st.session_state.memory

    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        max_tokens_limit=2000,
        verbose=True,
        combine_docs_chain_kwargs={
            "prompt": PROMPT,
            "document_prompt": DOCUMENT_PROMPT,
        },
    )

    return chain


# Generacion de respuesta
def answer_question(question, stream_handler, retrieval_handler):
    chain = get_chain()
    start = time.time()
    with get_openai_callback() as cb:
        result = chain(
            {
                "question": question,
            },
            callbacks=[stream_handler, retrieval_handler],
        )

        tokens = {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost_usd": cb.total_cost,
        }

        ic(result)
        ic(cb)
    end = time.time()

    answer = result["answer"]
    sources = result["source_documents"]

    add_to_db(
        st.session_state.chat_id,
        question,
        answer,
        tokens,
        end - start,
        st.session_state.model,
        st.session_state.message_id,
        sources,
    )


def main():
    st.set_page_config(
        page_title="Chatbot UNAP 📖",
        page_icon="🤖",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": "Chat capaz de responder preguntas relacionadas a reglamentos y documentos de la universidad Arturo Prat."
        },
    )

    show_pages_from_config()

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

    # Presentar documentos disponibles para consulta
    docs = get_repo_documents()
    with st.expander("Puedes realizar consultas sobre los siguientes documentos:"):
        for doc in docs:
            st.caption(doc.path.strip("documentos/").strip(".txt"))

    # Seleccion de modelo, ELIMINAR EN PRODUCCION
    chat_type = st.radio("Modelo", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"])

    # Set de preguntas predefinidas
    questions = [
        "¿Cuales son las tareas del decano?",
        "¿Que hago en caso de reprobar una asignatura?",
        "Explica en que consiste el trabajo de titulo",
        "¿Cuales son los requisitos para titularse?",
    ]
    qcol1, qcol2 = st.columns(2)
    ex_prompt = ""

    # Inicializacion variables de sesion
    if "msgs" not in st.session_state:
        st.session_state.msgs = StreamlitChatMessageHistory()
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            chat_memory=st.session_state.msgs,
            return_messages=True,
        )

    # Variables de sesion necesarias para firestore, ELIMINAR EN PRODUCCION
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(get_chats_len() + 1)
    if "message_id" not in st.session_state:
        st.session_state.message_id = str(
            get_messages_len(st.session_state.chat_id) + 1
        )
    if "model" not in st.session_state:
        st.session_state.model = chat_type

    if len(st.session_state.msgs.messages) == 0:
        st.session_state.msgs.add_ai_message("¡Hola! ¿en que puedo ayudarte?")

    avatars = {"human": "🧑‍💻", "ai": logo_path}
    for msg in st.session_state.msgs.messages:
        st.chat_message(msg.type, avatar=avatars[msg.type]).write(msg.content)

    for question in questions[:2]:
        with qcol1:
            if st.button(question, use_container_width=True):
                ex_prompt = question
    for question in questions[2:]:
        with qcol2:
            if st.button(question, use_container_width=True):
                ex_prompt = question

    # Input de usuario
    user_question = st.chat_input("Escribe tu pregunta...")

    if ex_prompt:
        user_question = ex_prompt
    if user_question or ex_prompt:
        st.chat_message("user", avatar="🧑‍💻").write(user_question)

        with st.chat_message("assistant", avatar=logo_path):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            answer = answer_question(user_question, stream_handler, retrieval_handler)

    if len(st.session_state.msgs.messages) > 0:
        streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="Proporciona feedback adicional (Opcional)",
            key=st.session_state.message_id,
            on_submit=update_feedback,
            args=(st.session_state.chat_id, st.session_state.message_id),
        )


if __name__ == "__main__":
    main()
