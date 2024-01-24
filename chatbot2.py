import os

import pinecone
import streamlit as st
from icecream import ic
from langchain.cache import InMemoryCache
from langchain.callbacks.manager import collect_runs
from langchain.globals import set_llm_cache
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langsmith import Client
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit_feedback import streamlit_feedback

from utils import config

os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner=False)
def get_llm():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        openai_api_key=config.OPENAI_API_KEY,
        max_tokens=1000,
        streaming=True,
    )
    return llm


@st.cache_resource(show_spinner=False)
def get_retriever():
    pinecone.init(
        api_key=config.PINECONE_API_KEY,
        environment=config.PINECONE_ENV,
    )
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    vectorstore = Pinecone.from_existing_index(
        index_name=config.PINECONE_INDEX_NAME, embedding=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    return retriever


@st.cache_resource(show_spinner=False)
def get_chain():
    template = """
    Eres un asistente para tareas de respuesta a preguntas. 
    Manten un tono conversacional, si el usuario te saluda, responde adecuadamente.
    Si la pregunta no es relevante para los documentos, responde 'Solo puedo responder preguntas sobre documentos de la universidad Arturo Prat.'
    
    Evita parrafos largos, responde mediante punteos y listas para facilitar la lectura.
    Siempre que respondas segun los documentos, cita el documento y el numero de articulo, donde corresponda.
    Genera tu respuesta en formato Markdown y utiliza footnotes para las referencias.
    
    Este es un ejemplo de como deberia verse una respuesta generada a partir de documentos:
        'El decano es el encargado de la administracion de la facultad. [^1]
        
        ### Referencias
        [^1]: Reglamento de la facultad de ingenieria, articulo 1.'
    
    Escribe el nombre del documento con un formato adecuado cuando lo cites.
    Sigue estas instrucciones y genera una respuesta para la pregunta.
    
    Utiliza los siguientes fragmentos de contexto recuperado para responder a la pregunta:
    {context}
    
    Pregunta: {question}
    Respuesta:
    """

    memory = st.session_state.memory
    retriever = get_retriever()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: memory.load_memory_variables(x)["chat_history"],
        }
    ).assign(answer=rag_chain)

    return rag_chain_with_source


@st.cache_resource(show_spinner=False)
def get_langsmith_client():
    client = Client(
        api_url=config.LANGCHAIN_ENDPOINT,
        api_key=config.LANGCHAIN_API_KEY,
    )
    return client


def process_chain_stream(prompt, sources_placeholder, response_placeholder):
    chain = get_chain()
    output = {}
    full_response = ""

    input_dict = {"question": prompt}
    with collect_runs() as cb:
        for chunk in chain.stream(prompt, config={"tags": ["Test Chat"]}):
            if "answer" in chunk:
                full_response += chunk["answer"]
                response_placeholder.markdown(full_response)

            if "context" in chunk:
                sources_placeholder.markdown("### 📚 Fuentes")
                sources_placeholder.markdown(
                    "La respuesta fue generada a partir de los siguientes textos:"
                )
                for doc in chunk["context"]:
                    file_name = doc.metadata["file_name"]
                    page_content = doc.page_content
                    sources_placeholder.caption(f"Extracto de **{file_name}**:")
                    sources_placeholder.code(page_content)

            for key in chunk:
                if key not in output:
                    output[key] = chunk[key]
                else:
                    output[key] += chunk[key]

            sources_placeholder.update(label="Respuesta generada", state="complete")
        st.session_state.run_id = cb.traced_runs[0].id
    st.session_state.memory.save_context(input_dict, {"answer": full_response})

    return full_response


if __name__ == "__main__":
    st.set_page_config(
        page_title="Chatbot UNAP 📖",
        page_icon="🤖",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": "Chat capaz de responder preguntas relacionadas a reglamentos y documentos de la universidad Arturo Prat."
        },
    )

    set_llm_cache(InMemoryCache())

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

    logo_path = "logos/unap_negativo.png"

    st.title("🤖 Chatbot UNAP")
    st.caption(
        "Este chatbot puede cometer errores. Si encuentras inexactitudes, reformula tu pregunta o consulta los documentos oficiales."
    )

    client = get_langsmith_client()

    if st.button("borrar cache"):
        get_chain.clear()
        get_llm.clear()
        get_retriever.clear()

    if "run_id" not in st.session_state:
        st.session_state.run_id = ""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            chat_memory=StreamlitChatMessageHistory(key="msgs"),
            return_messages=True,
        )

    avatars = {"human": "🧑‍💻", "ai": logo_path}
    for msg in st.session_state.msgs:
        st.chat_message(msg.type, avatar=avatars[msg.type]).write(msg.content)

    if prompt := st.chat_input(placeholder="Ask me a question!"):
        st.chat_message("user", avatar="🧑‍💻").write(prompt)
        with st.chat_message("assistant", avatar=logo_path):
            sources_placeholder = st.status("Consultando documentos")
            response_placeholder = st.empty()
            full_response = process_chain_stream(
                prompt, sources_placeholder, response_placeholder
            )
        response_placeholder.markdown(full_response)

    if len(st.session_state.msgs) > 0:
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="Proporciona feedback adicional (opcional):",
            key=f"feedback_{st.session_state.run_id}",
        )

    if st.session_state.run_id:
        score_mappings = {"thumbs": {"👍": 1, "👎": 0}}

        scores = score_mappings["thumbs"]

        if feedback:
            score = scores.get(feedback["score"])

            if score is not None:
                feedback_type_str = f"thumbs {feedback['score']}"

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
                st.warning("Invalid feedback score.")
