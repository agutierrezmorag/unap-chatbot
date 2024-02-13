import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langsmith import Client

from doc_manager.pinecone_management import get_retriever
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


# Funciones de agente
# Esta funcion no puede ser cacheada, para que funcione correctamente el agente
def get_agent():
    prompt = hub.pull("unap-chatbot/unap-rag-agent")

    document_prompt = PromptTemplate.from_template(
        "Nombre documento: {file_name} \nContenido: {page_content}"
    )
    doc_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Reglamentos"),
        "busqueda_reglamentos_unap",
        "Esta herramienta busca y recupera información sobre los reglamentos de la Universidad Arturo Prat. Úsala para encontrar reglas, pautas y procedimientos específicos de la universidad.",
        document_prompt=document_prompt,
    )

    wikipedia_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Wikipedia"),
        "busqueda_wikipedia_unap",
        "Esta herramienta busca y recupera información desde Wikipedia. Úsala para consultar la pagina de la Universidad Arturo Prat y obtener información general sobre la universidad, como su historia, funcionarios, ubicación, facultades y carreras.",
    )

    calendar_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Calendarios", k_results=2),
        "calendario_academico_unap",
        "Esta herramienta busca y recupera información sobre el calendario académico de la Universidad Arturo Prat. Úsala para encontrar fechas importantes, como el inicio y fin de semestres, días festivos, períodos de exámenes y otros eventos académicos.",
    )

    news_doc_prompt = PromptTemplate.from_template(
        "Titulo:{title} \nContenido: {page_content} \nFuente:{link} \nFecha de publicación: {publish_date}"
    )
    news_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Noticias"),
        "noticias_unap",
        "Esta herramienta busca y recupera noticias sobre la Universidad Arturo Prat. Úsala para encontrar actualizaciones recientes, anuncios y eventos relacionados con la universidad.",
        document_prompt=news_doc_prompt,
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
