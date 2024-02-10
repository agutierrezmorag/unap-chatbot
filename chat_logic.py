import os

import pinecone
import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Pinecone as pcvs
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client

from utils import config


def format_docs(docs):
    return "\n\n".join(
        f"<TITULO> {os.path.splitext(doc.metadata.get('file_name', doc.metadata.get('title')))[0]} </TITULO>\n<CONTENIDO> {doc.page_content} </CONTENIDO>"
        for doc in docs
    )


@st.cache_resource(show_spinner=False)
def get_langsmith_client():
    client = Client(
        api_url=config.LANGCHAIN_ENDPOINT,
        api_key=config.LANGCHAIN_API_KEY,
    )
    return client


# Para que funcione el agente, esta funcion no puede ser cacheada
# @st.cache_resource(show_spinner=False)
def get_llm():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0,
        max_tokens=1000,
        streaming=True,
    ).configurable_fields(
        model_name=ConfigurableField(
            id="gpt_model",
            name="GPT Model",
            description="The model to use for generating the response",
        )
    )
    return llm


@st.cache_resource(show_spinner=False)
def get_retriever(namespace="Reglamentos"):
    pc = pinecone.Pinecone(  # noqa: F841
        api_key=config.PINECONE_API_KEY,
        environment=config.PINECONE_ENV,
    )
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    try:
        vectorstore = pcvs.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=namespace,
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        wiki_vectorstore = pcvs.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace="Wikipedia",
        )
        wiki_retriever = wiki_vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        )

        ensembled_retrievers = EnsembleRetriever(
            retrievers=[retriever, wiki_retriever], weights=[0.2, 0.8]
        )

        return ensembled_retrievers
    except Exception as e:
        print(e)
        st.error(
            "Hubo un error al cargar el índice de documentos. Por favor, recarga la página y vuelve a intentarlo."
        )
        return None


@st.cache_resource(show_spinner=False)
def get_agent_retriever(namespace):
    pc = pinecone.Pinecone(  # noqa: F841
        api_key=config.PINECONE_API_KEY,
        environment=config.PINECONE_ENV,
    )
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    try:
        vectorstore = pcvs.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=namespace,
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        return retriever
    except Exception as e:
        print(e)
        st.error(
            "Hubo un error al cargar el índice de documentos. Por favor, recarga la página y vuelve a intentarlo."
        )
        return None


@st.cache_resource(show_spinner=False)
def get_chain():
    retriever = get_retriever()
    llm = get_llm()

    template = """
Eres un asistente cuyo encargo es responder preguntas sobre documentos institucionales. 
Mantén un tono amigable y conversacional. Si el usuario te saluda, responde adecuadamente. 
Si la pregunta no es relevante para los documentos o el historial de conversación, responde 'Solo puedo responder preguntas sobre documentos de la Universidad Arturo Prat.'
Evita párrafos largos. Utiliza listas y puntos para facilitar la lectura. Cuando respondas según los documentos, cita el documento y el número de artículo, si corresponde.
Genera tu respuesta en formato Markdown y utiliza notas al pie para las referencias. Aquí tienes un ejemplo de cómo debería verse una respuesta basada en documentos:
-------------
El decano es el encargado de la administración de la facultad. [^1]

### Referencias
[^1]: Reglamento de la facultad de ingeniería, articulo 1.
-------------
Escribe el nombre del documento con un formato adecuado cuando lo cites. 
Sigue estas instrucciones y genera una respuesta para la pregunta.
Pregunta: {input}
Utiliza los siguientes fragmentos de contexto recuperados para responder a la pregunta:
-------------
{context}
-------------
Respuesta:
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # Esta cadena de ejecución se encarga de combinar los documentos recuperados
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    # Esta cadena de ejecución se encarga de reformular la pregunta acorde al historial de conversación
    retriever_chain = create_history_aware_retriever(llm, retriever, rephrase_prompt)
    # Esta cadena de ejecución se encarga de dar respuesta a la pregunta, considerando las cadenas de ejecución anteriores
    retrieval_chain = create_retrieval_chain(retriever_chain, combine_docs_chain)

    return retrieval_chain


# Funciones de agente
# Esta funcion no puede ser cacheada, para que funcione correctamente el agente
def get_agent():
    template = """
Eres un asistente que ayuda a estudiantes a responder dudas sobre la Universidad Arturo Prat y sus reglamentos institucionales. No estas programado para responder preguntas sobre otros temas.

Tienes a tu disposición dos herramientas:
1. Una para buscar reglamentos de la Universidad Arturo Prat.
2. Otra para buscar información general sobre la universidad Arturo Prat.
3. Otra para buscar información sobre el calendario de la Universidad Arturo Prat.

Sigue estos pasos:
- Primero, utiliza la herramienta de búsqueda de documentos para encontrar informacion relevante para la pregunta.
- Utiliza el contenido de estos documentos para responder la pregunta.
- Si no puedes encontrar la respuesta en los documentos o necesitas información adicional, utiliza la herramienta de búsqueda de información general o de calendario.
- Utiliza toda la información reunida hasta ahora para responder la pregunta.
- Si luego de todos estos pasos aun no puedes responder la pregunta, responde indicando que no sabes la respuesta.

Considera lo siguiente:
- Siempre responde de forma formal, amigable y conversacional. Si el usuario te saluda, responde adecuadamente.
- Ignora las preguntas que no sean relevantes para los documentos o el historial de conversación.
- Genera tu respuesta en formato Markdown.

Pregunta: {question}
Respuesta:
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                template,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    document_prompt = PromptTemplate.from_template(
        "Nombre documento: {file_name} \nContenido: {page_content}"
    )

    doc_retriever_tool = create_retriever_tool(
        get_agent_retriever(namespace="Reglamentos"),
        "search_unap_documents",
        "Busca y devuelve información sobre los reglamentos de la Universidad Arturo Prat. Utilízalo para encontrar información relevante para dar respuesta a la pregunta.",
        document_prompt=document_prompt,
    )
    wikipedia_retriever_tool = create_retriever_tool(
        get_agent_retriever(namespace="Wikipedia"),
        "search_general_info_unap",
        "Busca y devuelve información general sobre la Universidad Arturo Prat. Utilízalo para encontrar información que no esté en los reglamentos.",
    )
    calendar_retriever_tool = create_retriever_tool(
        get_agent_retriever(namespace="Calendarios"),
        "check_unap_calendar",
        "Busca y devuelve información sobre el calendario de la Universidad Arturo Prat. Utilízalo para encontrar información sobre fechas importantes.",
    )
    news_retriever_tool = create_retriever_tool(
        get_agent_retriever(namespace="Noticias"),
        "check_unap_news",
        "Busca y devuelve información sobre noticias de la Universidad Arturo Prat. Utilízalo para encontrar información sobre artículos de eventos recientes.",
    )

    tools = [
        doc_retriever_tool,
        wikipedia_retriever_tool,
        calendar_retriever_tool,
        news_retriever_tool,
    ]

    agent = create_openai_tools_agent(get_llm(), tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=st.session_state.memory,
        max_iterations=3,
        max_execution_time=90.0,
        early_stopping_method="generate",
        return_intermediate_steps=True,
    ).with_config({"run_name": "Agent"})

    return agent_executor
