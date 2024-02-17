import pinecone
import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone
from langsmith import Client

from utils import config


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
        temperature=0.3,
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
def get_retriever(namespace: str, k_results: int = 3):
    pc = pinecone.Pinecone(  # noqa: F841
        api_key=config.PINECONE_API_KEY,
        environment=config.PINECONE_ENV,
    )
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    try:
        vectorstore = Pinecone.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=namespace,
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k_results}
        )

        return retriever
    except Exception as e:
        print(e)
        st.error(
            "Hubo un error al cargar el índice de documentos. Por favor, recarga la página y vuelve a intentarlo."
        )
        return None


@st.cache_resource(show_spinner=False)
def get_tools():
    document_prompt = PromptTemplate.from_template(
        "Nombre documento: {file_name} \nContenido: {page_content}"
    )
    doc_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Reglamentos"),
        "busqueda_reglamentos_unap",
        "Esta herramienta busca en la base de datos de documentos y reglamentos de la Universidad Arturo Prat. \
        Es ideal para encontrar información detallada sobre reglamentos, procedimientos, becas, lineamientos, etc. \
        Utilízala cuando necesites información específica y oficial de la universidad.",
        document_prompt=document_prompt,
    )

    wikipedia_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Wikipedia"),
        "busqueda_wikipedia_unap",
        "Esta herramienta busca información en Wikipedia. \
        Es útil para obtener una visión general de la universidad, como su historia, ubicación y fundación. \
        Utilízala cuando necesites información general y de fácil comprensión sobre la universidad.",
    )

    calendar_prompt = PromptTemplate.from_template(
        "Archivo: {file_name} \nContenido: {page_content}"
    )
    calendar_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Calendarios", k_results=2),
        "calendario_academico_unap",
        "Esta herramienta consulta los calendarios académicos de la Universidad Arturo Prat. \
        Es ideal para encontrar información sobre fechas importantes del año académico, como periodos de matrícula, inducción o vacaciones. \
        Utilízala cuando necesites conocer fechas o plazos específicos del calendario académico.",
        document_prompt=calendar_prompt,
    )

    news_doc_prompt = PromptTemplate.from_template(
        "Titulo:{title} \nContenido: {page_content} \nFuente:{link} \nFecha de publicación: {publish_date}"
    )
    news_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Noticias"),
        "noticias_unap",
        "Esta herramienta busca artículos publicados en el portal noticiero Actualidad UNAP. \
        Es útil para encontrar información sobre eventos, noticias y novedades recientes de la universidad. \
        Utilízala cuando necesites estar al día con los últimos acontecimientos en la universidad.",
        document_prompt=news_doc_prompt,
    )

    web_doc_prompt = PromptTemplate.from_template(
        "Contenido: {page_content} \nFuente: {source}"
    )
    web_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Web"),
        "web_unap",
        "Esta herramienta busca información en la página web de la Universidad Arturo Prat. \
        Es ideal para buscar información más especificada sobre la universidad, como las páginas de las facultades, carreras y los diversos departamentos. \
        Utilízala cuando necesites información que se encuentre en el sitio web oficial de la universidad.",
        document_prompt=web_doc_prompt,
    )

    tools = [
        doc_retriever_tool,
        wikipedia_retriever_tool,
        calendar_retriever_tool,
        news_retriever_tool,
        web_retriever_tool,
    ]
    return tools


@st.cache_resource(show_spinner=False)
def get_prompt():
    hub.pull("unap-chatbot/unap-rag-agent")


# Funciones de agente
# Esta funcion no puede ser cacheada, para que funcione correctamente el agente
def get_agent():
    prompt = get_prompt()
    llm = get_llm()
    tools = get_tools()

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=st.session_state.memory,
        max_iterations=5,
        max_execution_time=90.0,
        early_stopping_method="generate",
        return_intermediate_steps=True,
    ).with_config({"run_name": "Agent"})

    return agent_executor
