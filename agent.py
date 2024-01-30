import uuid

import streamlit as st
from icecream import ic
from langchain.cache import InMemoryCache
from langchain.callbacks.manager import collect_runs
from langchain.globals import set_llm_cache
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from st_pages import show_pages_from_config
from streamlit_feedback import streamlit_feedback

from chat_logic import get_agent, get_langsmith_client
from documents_manager import get_repo_documents
from utils import config


def agent_answer(prompt, response_placeholder):
    agent = get_agent()

    # Collect runs nos da el id del tracing en langsmith
    with collect_runs() as cb:
        try:
            full_response = agent.invoke(
                {"question": user_question},
                config={
                    "tags": [config.CHAT_ENVIRONMENT, st.session_state.model_type],
                    "metadata": {"user_session": st.session_state.session_id},
                },
            )

        except Exception as e:
            print(e)
            st.error(
                "Hubo un error al generar la respuesta. Por favor, recarga la página y vuelve a intentarlo."
            )
            return
        st.session_state.run_id = cb.traced_runs[0].id

    ic(full_response)
    return full_response


if __name__ == "__main__":
    st.set_page_config(
        page_title="Chatbot UNAP",
        page_icon="🤖",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": "Chat capaz de responder preguntas relacionadas a reglamentos y documentos de la universidad Arturo Prat."
        },
    )
    show_pages_from_config()

    # Eliminar el texto 'footnotes' generado por streamlit
    st.markdown(
        """
    <style>
    #footnotes {
        display: none
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Variables utiles a lo largo de la app
    set_llm_cache(InMemoryCache())
    logo_path = "logos/unap_negativo.png"
    client = get_langsmith_client()
    with st.sidebar:
        st.image(logo_path)

    st.title("🤖 Chatbot UNAP")
    st.caption(
        "Este chatbot puede cometer errores. Si encuentras inexactitudes, reformula tu pregunta o consulta los documentos oficiales."
    )

    # Lista de documentos disponibles para consultar
    docs = get_repo_documents()
    with st.expander("Puedes realizar consultas sobre los siguientes documentos:"):
        for doc in docs:
            st.caption(doc.path.strip("documentos/").strip(".txt"))

    # Inicializacion de variables de estado
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "run_id" not in st.session_state:
        st.session_state.run_id = ""
    if "msgs" not in st.session_state:
        st.session_state.msgs = StreamlitChatMessageHistory(key="msgs")
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            input_key="question",
            output_key="output",
            chat_memory=st.session_state.msgs,
            return_messages=True,
        )
    if "model_type" not in st.session_state:
        st.session_state.model_type = "gpt-3.5-turbo-1106"

    st.session_state.model_type = st.selectbox(
        "Tipo de modelo", ["gpt-3.5-turbo-1106", "gpt-4-turbo-preview"]
    )

    # Preguntas predefinidas
    questions = [
        "¿Cuales son las tareas del decano?",
        "¿Que hago en caso de reprobar una asignatura?",
        "Explica en que consiste el trabajo de titulo",
        "¿A que tipo de examenes puedo optar?",
    ]
    qcol1, qcol2 = st.columns(2)
    ex_prompt = ""
    for question in questions[:2]:
        with qcol1:
            if st.button(question, use_container_width=True):
                ex_prompt = question
    for question in questions[2:]:
        with qcol2:
            if st.button(question, use_container_width=True):
                ex_prompt = question

    # Historial de mensajes
    avatars = {"human": "🧑‍💻", "ai": logo_path}
    for msg in st.session_state.msgs.messages:
        st.chat_message(msg.type, avatar=avatars[msg.type]).write(msg.content)

    # Interaccion con el chatbot
    user_question = st.chat_input(placeholder="Escribe tu pregunta...")

    # Logica de respuesta
    if ex_prompt:
        user_question = ex_prompt
    if user_question:
        st.chat_message("user", avatar="🧑‍💻").write(user_question)
        with st.chat_message("assistant", avatar=logo_path):
            response_placeholder = st.container()
            full_response = agent_answer(user_question, response_placeholder)
        response_placeholder.markdown(full_response)

    # Botones de feedback
    if len(st.session_state.msgs.messages) > 0:
        feedback = streamlit_feedback(
            feedback_type="faces",
            optional_text_label="Proporciona feedback adicional (opcional):",
            key=f"feedback_{st.session_state.run_id}",
        )

    # Registro de feedback
    if st.session_state.run_id:
        score_mappings = {
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0}
        }
        scores = score_mappings["faces"]

        if feedback:
            score = scores.get(feedback["score"])

            if score is not None:
                feedback_type_str = f"faces {feedback['score']}"

                feedback_record = client.create_feedback(
                    st.session_state.run_id,
                    feedback_type_str,
                    score=score,
                    comment=feedback.get("text"),
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
            else:
                st.warning("Puntuación no válida. No se registró el feedback.")