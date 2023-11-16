import os
import pinecone
import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
HUGGINGFACEHUB_API_TOKEN = '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'
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())

llm = ChatOpenAI(model='gpt-3.5-turbo')

# Listado de nombres de documentos
@st.cache_data
def get_doc_names():
    file_names = []
    for file in os.listdir("documentos"):
        file_names.append(file)
    return file_names


@st.cache_resource
def load_and_split_docs():
    # Carga de documentos
    raw_text_files = []
    for file in os.listdir("documentos"):
            text_path = "./documentos/" + file
            loader = TextLoader(text_path, encoding='UTF-8')
            raw_text_files.extend(loader.load())

    # Split de textos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=128,
        length_function=len
        )
    texts = text_splitter.split_documents(raw_text_files)
    return texts


# Embedding
## Importar vectorstore
@st.cache_resource
def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY, 
        environment=PINECONE_ENV,
    )
    index_name = "chatbot-unap"

    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
    return vectorstore


## Crear vectorstore
def do_embedding(text_chunks):
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY, 
        environment=PINECONE_ENV,
    )
    index_name = "chatbot-unap"

    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
    vectorstore = Pinecone.from_documents(text_chunks, embeddings, index_name=index_name)
    return vectorstore


def answer_question(vectorstore, question):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, max_tokens_limit=2000, return_source_documents=True)

    with get_openai_callback() as cb:
        result = chain({"question": question, "chat_history": st.session_state.chat_history})
        print(result['source_documents'])
        tokens_used = cb.total_tokens
        print(cb)

    st.session_state.chat_history = [(question, result['answer'])]

    return result['answer'], tokens_used


def main():
    load_dotenv()
    st.set_page_config(
        page_title="UNAP Chatbot 📖",
        page_icon="🤖",
    )

    vectorstore = get_vectorstore()
    file_names = get_doc_names()

    with st.sidebar:
        if st.button('Crear embedding 📖'):
            do_embedding(load_and_split_docs())
        st.write(file_names)

    st.title('UNAP Chatbot 🤖📖')
    st.write('Chat capaz de responder preguntas relacionadas a reglamentos y documentos relacionados con la universidad Arturo Prat. Actualmente es consciente de', len(file_names), 'documentos.')
    
    # # Inicializacion historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Mantener historial en caso de rerun de app
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Escribe tu pregunta..."):
        # Agregar input de usuario al hisotrial
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Mostrar input en su contenedor
        with st.chat_message("user"):
            st.markdown(prompt)
        # Mostrar respuesta del LLM
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner('Generando respuesta...'):
                full_response, tokens = answer_question(vectorstore=vectorstore, question=prompt)
            message_placeholder.markdown(f'{full_response} **({tokens})**')
        # Agregar respuesta del LLM al historial
        st.session_state.messages.append({"role": "assistant", "content": f'{full_response} {tokens}'})
    

if __name__ == '__main__':
    main()