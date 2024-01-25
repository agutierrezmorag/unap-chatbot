import pinecone
import streamlit as st
from langchain import hub
from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client

from utils import config


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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
        model="gpt-3.5-turbo-1106",
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.7,
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

    prompt = hub.pull("unap-chatbot/unap-chatbot-rag")

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: x["context"]))
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
