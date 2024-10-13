import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langsmith import Client

from doc_manager.pinecone_management import get_or_create_vectorstore
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
        model_name=config.OPENAI_AI_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.7,
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
def get_retriever(namespace: str, k_results: int = 5):
    vectorstore = get_or_create_vectorstore(namespace)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k_results}
    )

    return retriever


@st.cache_resource(show_spinner=False)
def get_tools():
    document_prompt = PromptTemplate.from_template(
        "Nombre del documento: {file_name} \nContenido: {page_content}"
    )
    doc_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Reglamentos"),
        "buscador_reglamentos_unap",
        "Este es el principal recurso de información. Busca en la base de datos de reglamentos de la Universidad Arturo Prat. \
        Proporciona información detallada y oficial sobre reglamentos, procedimientos, becas y directrices. Utilízalo para corroborar información oficial.",
        document_prompt=document_prompt,
    )

    wikipedia_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Wikipedia"),
        "wikipedia_unap",
        "Este recurso proporciona información general de Wikipedia sobre la Universidad Arturo Prat, incluyendo historia, ubicación y sedes. \
        Úsalo como un recurso secundario para obtener una visión general.",
    )

    calendar_prompt = PromptTemplate.from_template(
        "Archivo: {file_name} \nContenido: {page_content}"
    )
    calendar_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Calendarios"),
        "calendario_academico_unap",
        "Consulta los calendarios académicos de la Universidad Arturo Prat. \
        Proporciona fechas importantes del año académico, como periodos de matrícula, inducción y vacaciones.",
        document_prompt=calendar_prompt,
    )

    news_doc_prompt = PromptTemplate.from_template(
        "Título:{title} \nContenido: {page_content} \nFuente:{link} \nFecha de publicación: {publish_date}"
    )
    news_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Noticias"),
        "actualidad_unap",
        "Recupera artículos del portal de noticias Actualidad UNAP. \
        Útil para mantenerse al día con eventos, noticias y desarrollos recientes en la universidad.",
        document_prompt=news_doc_prompt,
    )

    web_doc_prompt = PromptTemplate.from_template(
        "Contenido: {page_content} \nFuente: {source}"
    )
    web_retriever_tool = create_retriever_tool(
        get_retriever(namespace="Web"),
        "sitio_web_unap",
        "Busca información en el sitio web oficial de la Universidad Arturo Prat. \
        Útil para obtener detalles específicos sobre facultades, carreras y departamentos.",
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
def question_suggester_chain():
    llm = get_llm()
    prompt = hub.pull("unap-chatbot/unap-chatbot-q-suggester")

    chain = (
        {"input": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    ).with_config({"run_name": "Question Suggester"})
    return chain


@st.cache_resource(show_spinner=False)
def get_agent_prompt():
    return hub.pull("unap-chatbot/unap-rag-agent")


# Funciones de agente
# Esta funcion no puede ser cacheada, para que funcione correctamente el agente
def get_agent():
    prompt = get_agent_prompt()
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
