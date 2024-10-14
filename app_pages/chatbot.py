import asyncio
import random
import time

import streamlit as st
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.tracers.context import collect_runs
from streamlit_feedback import streamlit_feedback
from termcolor import cprint

from chat_logic import get_agent, get_langsmith_client, question_suggester_chain
from document_management.github_management import get_repo_docs_as_pd
from utils import config


async def agent_answer(prompt, agent_thoughts_placeholder, response_placeholder):
    agent = get_agent()
    suggester_llm = question_suggester_chain()
    full_response = ""
    full_output = ""

    with collect_runs() as cb:
        try:
            async for event in agent.astream_events(
                {"input": prompt},
                config={
                    "tags": [config.CHAT_ENVIRONMENT],
                    "metadata": {"conversation_id": st.session_state.session_id},
                },
                version="v1",
            ):
                kind = event["event"]
                if kind == "on_chain_end":
                    if event["name"] == "Agent":
                        agent_thoughts_placeholder.markdown("- 🎉 Ya sé la respuesta.")
                        time.sleep(1)
                        agent_thoughts_placeholder.update(
                            label="🤗 Respuesta generada.",
                            expanded=False,
                            state="complete",
                        )
                        suggested_question = suggester_llm.invoke(
                            st.session_state.memory.buffer[-2:]
                        )
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                elif kind == "on_tool_start":
                    event_name = event["name"]
                    query = event["data"].get("input")["query"]
                    if event_name == "buscador_reglamentos_unap":
                        agent_thoughts_placeholder.markdown(
                            f"- 📚 Consultando **{query}** en los reglamentos..."
                        )
                    if event_name == "calendario_academico_unap":
                        agent_thoughts_placeholder.markdown(
                            f"- 📅 Consultando **{query}** en el calendario académico..."
                        )
                    if event_name == "actualidad_unap":
                        agent_thoughts_placeholder.markdown(
                            f"- 📰 Consultando **{query}** en las noticias..."
                        )
                    if event_name == "sitio_web_unap":
                        agent_thoughts_placeholder.markdown(
                            f"- 🔗 Revisando **{query}** en la pagina web..."
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
                            "- ✨ Creo haber encontrado textos relevantes:"
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
                "Hubo un error al generar la respuesta. Por favor, vuelve a intentarlo."
            )
            return
        st.session_state.run_id = cb.traced_runs[0].id

    if full_response:
        st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
        st.button(
            f"✨ {suggested_question}",
            on_click=submit_question,
            args=(suggested_question,),
        )
    st.session_state.user_question = ""


def submit_question(question):
    st.session_state.user_question = question


if __name__ == "__page__":
    st.set_page_config(
        page_title="Chatbot UNAP",
        page_icon="🤖",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": "Chat capaz de responder preguntas relacionadas a reglamentos y documentos de la universidad Arturo Prat."
        },
    )

    # CSS necesario para mostrar el boton de sugerencia de preguntas como texto plano
    st.markdown(
        """
    <style>
    .element-container:has(style){
        display: none;
    }
    #button-after {
        display: none;
    }
    .element-container:has(#button-after) {
        display: none;
    }
    .element-container:has(#button-after) + div button {
        border: none;
        background: none;
        font-style: italic;
        text-align: left;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Variables utiles a lo largo de la app
    set_llm_cache(InMemoryCache())
    logo_path = "logos/unap_negativo.png"
    client = get_langsmith_client()

    # Lista de documentos disponibles para consultar
    docs = get_repo_docs_as_pd("Reglamentos")
    with st.expander("¿Sobre qué puedo preguntar?"):
        st.markdown(
            "- 📰 Noticias publicadas en [Actualidad UNAP](https://www.unap.cl/prontus_unap/site/edic/base/port/actualidad.html)"
        )
        st.markdown("- 🗓️ Los calendarios académicos")
        st.markdown(
            "- 🔗 Contenido disponible en la [página web de la universidad](https://www.unap.cl/) y [admision UNAP](https://www.admisionunap.cl/)"
        )
        st.markdown(
            "- 📃 Reglamentos y documentos generales de la universidad, como los siguientes:"
        )
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
        st.caption(
            "Este chatbot puede cometer errores. Si encuentras inexactitudes, reformula tu pregunta o consulta los documentos oficiales."
        )

    # Preguntas predefinidas
    question_pool = [
        "¿Cuáles son las tareas del decano?",
        "¿Qué hago en caso de reprobar una asignatura?",
        "Explica en qué consiste el trabajo de título",
        "Cuéntame la historia de la universidad",
        "¿Qué es el levantamiento de pre-requisito?",
        "¿Qué funciones cumple el consejo de carrera?",
        "Cuéntame sobre algunas noticias recientes",
        "¿Cuanto dura el primer semestre académico?",
        "¿La UNAP ofrece tutorias?",
    ]
    questions = random.sample(question_pool, 4)
    qcol1, qcol2 = st.columns(2)
    ex_prompt = ""
    for question in questions[:2]:
        with qcol1:
            st.button(
                question,
                use_container_width=True,
                on_click=submit_question,
                args=(question,),
            )
    for question in questions[2:]:
        with qcol2:
            st.button(
                question,
                use_container_width=True,
                on_click=submit_question,
                args=(question,),
            )

    # Historial de mensajes
    avatars = {"human": "🧑‍💻", "ai": logo_path}
    for msg in st.session_state.msgs.messages:
        st.chat_message(msg.type, avatar=avatars[msg.type]).write(msg.content)

    # Interaccion con el chatbot
    if question := st.chat_input(placeholder="Escribe tu pregunta..."):
        submit_question(question)

    if st.session_state.user_question != "":
        st.chat_message("user", avatar="🧑‍💻").write(st.session_state.user_question)
        with st.chat_message("assistant", avatar=logo_path):
            response_placeholder = st.empty()
            agent_thoughts_placeholder = st.status("🤔 Pensando...", expanded=False)
            asyncio.run(
                agent_answer(
                    st.session_state.user_question,
                    agent_thoughts_placeholder,
                    response_placeholder,
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
