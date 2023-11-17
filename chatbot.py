import os, json
import pinecone
import streamlit as st
from dotenv import load_dotenv
from google.cloud import firestore
import time

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

from langchain.prompts import PromptTemplate

# API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

load_dotenv()
set_llm_cache(InMemoryCache())


# Instanciar llm
@st.cache_resource(show_spinner=False)
def get_llm():
    llm = ChatOpenAI(model='gpt-3.5-turbo', max_tokens=1000)
    return llm


# Conectar con firestore
@st.cache_resource
def db_connection():
    key_dict = json.loads(st.secrets["textkey"])
    creds = firestore.Client.service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds, project="streamlit-reddit")
    return db


# Total de chats
def get_chats_len():
    chats_ref = db_connection().collection('chats').get()
    return len(chats_ref)


# Listado de nombres de documentos
@st.cache_data
def get_doc_names():
    file_names = []
    for file in os.listdir("documentos"):
        file_names.append(file)
    return file_names


# Carga y split de textos
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


# Embeddings
# Importar vectorstore
@st.cache_resource(show_spinner=False)
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


# Generacion de respuesta
def answer_question(vectorstore, question):
    template = """
    You are a helpful chatbot, always eager to answer questions made by students and workers of a college.
    Your job is to answer questions based on documents and their context, and improve your answers from previous answers.
    Don't try to make up an answer, if you don't know, just say that you don't know.
    Answer in the same language the question was asked. Always answer formally. 
    If it takes you more than 5 seconds to answer, just say you can not answer right now and to ask again later.

    Context: {context}

    Question: {question}
    Answer: 
    """
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
    chain = ConversationalRetrievalChain.from_llm(llm=get_llm(), retriever=retriever, max_tokens_limit=2000, verbose=True, combine_docs_chain_kwargs={"prompt": PROMPT})

    with get_openai_callback() as cb:
        result = chain({"question": question, "chat_history": st.session_state.chat_history})
        with st.expander('tokens'):
            cb
        print(result)
        tokens = {
            'total_tokens': cb.total_tokens,
            'prompt_tokens': cb.prompt_tokens,
            'completion_tokens': cb.completion_tokens,
            'total_cost_usd': cb.total_cost
        }

    st.session_state.chat_history = [(question, result['answer'])]
    return result['answer'], tokens


# Registrar datos en la base de datos 
def add_to_db(question, answer, tokens, time_to_answer):
    chat_id = st.session_state.chat_id
    db = db_connection()

    chats_ref = db.collection('chats')
    chat_doc_ref = chats_ref.document(chat_id)

    # Revisar si documento con chat_id existe
    chat_doc = chat_doc_ref.get()
    if not chat_doc.exists:
        # Crearlo en caso de que no exista
        chat_doc_ref.set({})

    # Agregar pregunta y respuesta a sub coleccion messages
    messages_ref = chat_doc_ref.collection('messages')
    messages_ref.add({
        'question': question,
        'answer': answer,
        'tokens': tokens,
        'time_to_answer': time_to_answer
    })


def main():
    st.set_page_config(
        page_title="UNAP Chatbot 📖",
        page_icon="🤖",
    )

    file_names = get_doc_names()
    col1,col2,col3 = st.columns([3,0.5,0.5])

    st.title('🤖 UNAP Chatbot 📖')
    st.write('Chat capaz de responder preguntas relacionadas a reglamentos y documentos relacionados con la universidad Arturo Prat. Actualmente es consciente de', 
             len(file_names), 
             'documentos.'
             )
    
    with st.expander('Listado de documentos'):
        st.write(file_names)

    # # Inicializacion historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! ¿Como te puedo ayudar?"}]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "ask_feedback" not in st.session_state:
        st.session_state.ask_feedback = True
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(get_chats_len() + 1)

    # Mantener historial en caso de rerun de app
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Escribe tu pregunta...")

    if prompt:
        # Agregar input de usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Mostrar input en su contenedor
        with st.chat_message("user"):
            st.markdown(prompt)
        # Mostrar respuesta del LLM
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            start = time.time()
            with st.spinner('Generando respuesta...'):
                full_response, tokens = answer_question(vectorstore=get_vectorstore(), question=prompt)
            message_placeholder.markdown(full_response + '▌')
            end = time.time()
            if st.session_state.ask_feedback:
                with col1:
                    st.markdown('*¿Que te parecio la velocidad de la respuesta?*')
                with col2:
                    st.button(":thumbsup:", key='like')
                with col3:
                    st.button(":thumbsdown:", key='dislike')
        # Agregar respuesta del LLM al historial
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        add_to_db(question=prompt, answer=full_response, tokens=tokens, time_to_answer=end-start)


if __name__ == '__main__':
    main()
