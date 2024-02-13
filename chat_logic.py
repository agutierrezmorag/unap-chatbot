import pinecone
import streamlit as st
from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Pinecone as pcvs
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
def get_retriever():
    pc = pinecone.Pinecone(  # noqa: F841
        api_key=config.PINECONE_API_KEY,
        environment=config.PINECONE_ENV,
    )
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    try:
        doc_vectorstore = pcvs.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace="Reglamentos",
        )
        doc_retriever = doc_vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        wiki_vectorstore = pcvs.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace="Wikipedia",
        )
        wiki_retriever = wiki_vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        news_vectorstore = pcvs.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace="Noticias",
        )
        news_retriever = news_vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        calendar_vectorstore = pcvs.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace="Calendarios",
        )
        calendar_retriever = calendar_vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        ensembled_retrievers = EnsembleRetriever(
            retrievers=[
                doc_retriever,
                wiki_retriever,
                news_retriever,
                calendar_retriever,
            ]
        )

        return ensembled_retrievers
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
