import os
import pinecone
import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


# API keys
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


# Listado de nombres de documentos
def get_doc_names():
    file_names = []
    for file in os.listdir("documentos"):
        file_names.append(file)
    return file_names


# Embedding
@st.cache_resource
def create_or_get_embeddings():
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

    # Creacion o pull de index de embedding
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY, 
        environment=PINECONE_ENV,
    )
    index_name = "unap-chatbot"

    # en caso de no existir el index, se crea
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
        docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    # de lo contrario, se hace un pull al index existente
    else:
        docsearch = Pinecone.from_existing_index(index_name, embeddings)

    return (texts, docsearch)


# LLM
def llm_load():
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm=llm, chain_type='stuff')
    return chain


# Paso de user input a llm
def answer(docsearch, query, chain):
     docs = docsearch.similarity_search(query)
     return chain.run(input_documents=docs, question=query)


def main():
    st.set_page_config(
        page_title="UNAP Chatbot 📖",
        page_icon="🤖",
    )

    # Variables a utilizar
    file_names = get_doc_names();
    text_files, docsearch = create_or_get_embeddings();
    chain = llm_load();

    st.title('UNAP Chatbot 🤖📖')
    st.write('Chat capaz de responder preguntas relacionadas a reglamentos y documentos relacionados con la universidad Arturo Prat. Actualmente es consciente de', len(file_names), 'documentos.')
    with st.expander('Listado de documentos'):
         st.write(file_names)

    # Inicializacion historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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
                full_response = answer(docsearch=docsearch, query=prompt, chain=chain)
            message_placeholder.markdown(full_response + "▌")
        # Agregar respuesta del LLM al historial
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()