import asyncio
import time
import uuid

import streamlit as st
from langchain.cache import InMemoryCache
from langchain.callbacks.manager import collect_runs
from langchain.globals import set_llm_cache
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from st_pages import show_pages_from_config
from streamlit_feedback import streamlit_feedback
from termcolor import cprint

from chat_logic import get_agent, get_langsmith_client
from doc_manager.github_management import get_repo_docs_as_pd
from utils import config


async def agent_answer(prompt, agent_thoughts_placeholder, response_placeholder):
    agent = get_agent()
    full_response = ""
    full_output = ""

    with collect_runs() as cb:
        try:
            async for event in agent.astream_events(
                {"question": user_question},
                config={
                    "tags": [config.CHAT_ENVIRONMENT, st.session_state.model_type],
                    "metadata": {"user_session": st.session_state.session_id},
                },
                version="v1",
            ):
                kind = event["event"]
                if kind == "on_chain_end":
                    if event["name"] == "Agent":
                        time.sleep(1)
                        agent_thoughts_placeholder.update(
                            label="🤗 Respuesta generada.",
                            expanded=False,
                            state="complete",
                        )
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                elif kind == "on_tool_start":
                    event_name = event["name"]
                    query = event["data"].get("input")["query"]
                    if event_name == "search_unap_documents":
                        agent_thoughts_placeholder.markdown(
                            f"- 📚 Consultando **{query}** en los reglamentos..."
                        )
                    if event_name == "check_unap_calendar":
                        agent_thoughts_placeholder.markdown(
                            f"- 📅 Consultando **{query}** en el calendario académico..."
                        )
                    else:
                        agent_thoughts_placeholder.markdown(
                            f"- 🔍 Consultando **{query}** en los documentos..."
                        )
                elif kind == "on_tool_end":
                    event_name = event["name"]
                    output = event["data"].get("output")
                    if output:
                        agent_thoughts_placeholder.markdown(
                            "- 🤓 Creo haber encontrado textos relevantes:"
                        )
                        full_output += output
                        agent_thoughts_placeholder.text_area(
                            "Contexto",
                            help="La IA utiliza este contexto para generar la respuesta. \
                                Estos textos provienen de una variedad de reglamentos y documentos generales de la universidad.",
                            value=full_output,
                            disabled=True,
                        )

        except Exception as e:
            cprint(e, "red")
            st.error(
                "Hubo un error al generar la respuesta. Por favor, recarga la página y vuelve a intentarlo."
            )
            return
        st.session_state.run_id = cb.traced_runs[0].id

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

    st.caption(
        "Este chatbot puede cometer errores. Si encuentras inexactitudes, reformula tu pregunta o consulta los documentos oficiales."
    )

    # Lista de documentos disponibles para consultar
    docs = get_repo_docs_as_pd("txt")
    with st.expander("Puedes realizar consultas sobre los siguientes documentos:"):
        st.dataframe(
            docs,
            height=200,
            use_container_width=True,
            hide_index=True,
            column_order=["name", "download_url"],
            column_config={
                "name": st.column_config.Column("📄 Nombre documento", width="large"),
                "download_url": st.column_config.LinkColumn(
                    "⬇️ Descarga",
                    display_text="Descargar",
                    width="small",
                ),
            },
        )

    # Inicializacion de variables de estado
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
        "¿Cuáles son las tareas del decano?",
        "¿Qué hago en caso de reprobar una asignatura?",
        "Explica en qué consiste el trabajo de título",
        "¿A qué tipo de exámenes puedo optar?",
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
            response_placeholder = st.empty()
            agent_thoughts_placeholder = st.status("🤔 Pensando...", expanded=True)
            full_response = asyncio.run(
                agent_answer(
                    user_question, agent_thoughts_placeholder, response_placeholder
                )
            )

    # Botones de feedback
    if len(st.session_state.msgs.messages) > 0:
        feedback = streamlit_feedback(
            feedback_type="faces",
            optional_text_label="Proporciona retroalimentación adicional (opcional):",
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
