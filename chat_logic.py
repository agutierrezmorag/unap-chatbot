import os

import pinecone
import streamlit as st
from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    ConfigurableField,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client

from utils import config


def format_docs(docs):
    return "\n\n".join(
        f"{os.path.splitext(doc.metadata['file_name'])[0]}: {doc.page_content}"
        for doc in docs
    )


@st.cache_resource(show_spinner=False)
def get_langsmith_client():
    client = Client(
        api_url=config.LANGCHAIN_ENDPOINT,
        api_key=config.LANGCHAIN_API_KEY,
    )
    return client


@st.cache_resource(show_spinner=False)
def get_llm():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",
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
def get_retriever():
    pinecone.init(
        api_key=config.PINECONE_API_KEY,
        environment=config.PINECONE_ENV,
    )
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    try:
        vectorstore = Pinecone.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME, embedding=embeddings
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
    except Exception as e:
        print(e)
        st.error(
            "Hubo un error al cargar el índice de documentos. Por favor, recarga la página y vuelve a intentarlo."
        )
    return retriever


@st.cache_resource(show_spinner=False)
def get_chain():
    memory = st.session_state.memory
    retriever = get_retriever()
    llm = get_llm()

    template = """
Eres un asistente cuyo encargo es responder preguntas sobre documentos institucionales. Se amigable y mantén un tono conversacional, si el usuario te saluda, responde adecuadamente.
Si la pregunta no es relevante para los documentos o el historial de conversación, responde 'Solo puedo responder preguntas sobre documentos de la universidad Arturo Prat.'
Evita párrafos largos, utiliza punteo y listas para facilitar la lectura. Siempre que respondas según los documentos, cita el documento y el numero de articulo, donde corresponda.
Genera tu respuesta en formato Markdown y utiliza footnotes para las referencias. Este es un ejemplo de como debería verse una respuesta generada a partir de documentos:
-------------
El decano es el encargado de la administración de la facultad. [^1]

### Referencias
[^1]: Reglamento de la facultad de ingeniería, articulo 1.
-------------
Escribe el nombre del documento con un formato adecuado cuando lo cites.
Sigue estas instrucciones y genera una respuesta para la pregunta.
Pregunta: {question}
Utiliza los siguientes fragmentos de contexto recuperado para responder a la pregunta:
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
