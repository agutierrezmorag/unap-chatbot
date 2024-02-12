import os

import pinecone
import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.tools.wikipedia import Wikipedia
from langchain_community.vectorstores import Pinecone as pcvs
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client

from utils import config
from utils.wikipedia_retriever import CustomWikipediaRetriever


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

    prompt = hub.pull("unap-chatbot/unap-rag-chat_model")

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
    prompt = hub.pull("unap-chatbot/unap-rag-agent")

    document_prompt = PromptTemplate.from_template(
        "Nombre documento: {file_name} \nContenido: {page_content}"
    )

    doc_retriever_tool = create_retriever_tool(
        get_agent_retriever(namespace="Reglamentos"),
        "busqueda_reglamentos_unap",
        "Esta herramienta busca y recupera información sobre los reglamentos de la Universidad Arturo Prat. Úsala para encontrar reglas, pautas y procedimientos específicos de la universidad.",
        document_prompt=document_prompt,
    )
    wikipedia_retriever_tool = create_retriever_tool(
        CustomWikipediaRetriever(
            page_name="Universidad Arturo Prat",
            lang="es",
            load_max_docs=1,
            top_k_results=1,
            doc_content_chars_max=1000,
        ),
        "busqueda_wikipedia_unap",
        "Esta herramienta busca y recupera información desde Wikipedia. Úsala para consultar la pagina de la Universidad Arturo Prat y obtener información general sobre la universidad, como su historia, ubicación, facultades y carreras.",
    )
    calendar_retriever_tool = create_retriever_tool(
        get_agent_retriever(namespace="Calendarios"),
        "calendario_academico_unap",
        "Esta herramienta busca y recupera información sobre el calendario académico de la Universidad Arturo Prat. Úsala para encontrar fechas importantes, como el inicio y fin de semestres, días festivos y períodos de exámenes.",
    )
    news_retriever_tool = create_retriever_tool(
        get_agent_retriever(namespace="Noticias"),
        "noticias_unap",
        "Esta herramienta busca y recupera noticias sobre la Universidad Arturo Prat. Úsala para encontrar actualizaciones recientes, anuncios y eventos relacionados con la universidad.",
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
