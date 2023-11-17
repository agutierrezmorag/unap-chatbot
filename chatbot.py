import os
import pinecone
import streamlit as st
import streamlit_analytics as st_analytics
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

# API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

load_dotenv()
set_llm_cache(InMemoryCache())

llm = ChatOpenAI(model='gpt-3.5-turbo', max_tokens=1000)

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
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, max_tokens_limit=2000, verbose=True)

    with get_openai_callback() as cb:
        result = chain({"question": question, "chat_history": st.session_state.chat_history})
        print(result)
        tokens_used = cb.total_tokens
        print(cb)

    st.session_state.chat_history = [(question, result['answer'])]

    return result['answer'], tokens_used


def main():
    st.set_page_config(
        page_title="UNAP Chatbot 📖",
        page_icon="🤖",
    )

    vectorstore = get_vectorstore()
    file_names = get_doc_names()

    st.title('UNAP Chatbot 🤖📖')
    st.write('Chat capaz de responder preguntas relacionadas a reglamentos y documentos relacionados con la universidad Arturo Prat. Actualmente es consciente de', len(file_names), 'documentos.')
    
    with st.expander('Listado de documentos'):
        st.write(file_names)

    # # Inicializacion historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! ¿Como te puedo ayudar?"}]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False

    st_analytics.start_tracking()
    # Mantener historial en caso de rerun de app
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Escribe tu pregunta..."):
        # Agregar input de usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Mostrar input en su contenedor
        with st.chat_message("user"):
            st.markdown(prompt)
        # Mostrar respuesta del LLM
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner('Generando respuesta...'):
                full_response, tokens = answer_question(vectorstore=vectorstore, question=prompt)
            message_placeholder.markdown(f'{full_response} *({tokens})*')
            col1,col2,col3 = st.columns([3,0.5,0.5])
            if not st.session_state.feedback_given:
                with col1:
                    st.markdown('*¿Que te parecio la velocidad de la respuesta?*')
                with col2:
                    st.button(":thumbsup:")
                with col3:
                    st.button(":thumbsdown:")
            else:
                with col1:
                    st.markdown('*Gracias por el feedback!*')
        # Agregar respuesta del LLM al historial
        st.session_state.messages.append({"role": "assistant", "content": f'{full_response} *({tokens})*'})
    st_analytics.stop_tracking()

if __name__ == '__main__':
    main()