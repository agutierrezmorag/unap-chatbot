import os

import pinecone
import streamlit as st
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.retrievers import EnsembleRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Pinecone as pcvs
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableField, RunnableMap, RunnablePassthrough
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
def get_agent_retriever(namespace="Reglamentos"):
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
    memory = st.session_state.memory
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
Pregunta: {question}
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
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableMap(
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: memory.load_memory_variables(x)["chat_history"],
        }
    ).assign(answer=rag_chain)

    return rag_chain_with_source


# Funciones de agente
# Esta funcion no puede ser cacheada, para que funcione correctamente el agente
def get_agent():
    template = """
Eres un asistente que ayuda a estudiantes a responder dudas sobre la Universidad Arturo Prat y sus reglamentos institucionales. Tienes a tu disposición dos herramientas:

1. Una para buscar documentos de reglamentos.
2. Otra para buscar en Wikipedia.

Sigue estos pasos:

- Primero, utiliza la herramienta de búsqueda de documentos para encontrar documentos relevantes a la pregunta.
- Si no puedes encontrar la respuesta en los documentos, utiliza la herramienta de búsqueda en Wikipedia.
- Solo responde preguntas que tengan relación con la universidad o el historial de conversación, ignorando las que no tengan relación.
- Si luego de todos estos pasos aun no puedes responder la pregunta, responde indicando que no sabes la respuesta.

Recuerda:

- Siempre responde de forma formal, amigable y conversacional. Si el usuario te saluda, responde adecuadamente.
- Cuando respondas según los documentos, cita el documento y el número de artículo. Si no sabes el número de artículo, puedes omitir la cita. NO cites solo con el nombre del documento.

Genera tu respuesta en formato Markdown y utiliza notas al pie para las referencias.

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

    doc_retriever_tool = create_retriever_tool(
        get_agent_retriever(),
        "search_unap_documents",
        "Searches and returns excerpts from UNAP documents. Use it to find relevant documents to answer a question.",
    )
    wikipedia_retriever_tool = create_retriever_tool(
        get_agent_retriever(namespace="Wikipedia"),
        "search_wikipedia",
        "Searches and returns excerpts from UNAP's Wikipedia page. Use it to find relevant information about UNAP that is not available in the documents.",
    )

    tools = [doc_retriever_tool, wikipedia_retriever_tool]

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
