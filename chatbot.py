import os
import json
import time
import pinecone
import streamlit as st
from dotenv import load_dotenv
from google.cloud import firestore
from google.oauth2 import service_account

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

# from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent

from trubrics.integrations.streamlit import FeedbackCollector
from streamlit_feedback import streamlit_feedback


# API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')

load_dotenv()
set_llm_cache(InMemoryCache())


# Instanciar llm
@st.cache_resource(show_spinner=False)
def get_llm():
    model = st.session_state.model
    llm = ChatOpenAI(model=model, max_tokens=1000)
    return llm


# Conectar con firestore
@st.cache_resource
def db_connection():
    key_dict = json.loads(st.secrets["textkey"])
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db


# Total de chats
def get_chats_len():
    chats_ref = db_connection().collection('chats').get()
    return len(chats_ref)


def get_messages_len():
    chat_id = st.session_state.chat_id
    message_ref = db_connection().collection(
        'chats').document(chat_id).collection('messages').get()
    return len(message_ref)


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


# Crear vectorstore
def do_embedding(text_chunks):
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
    )
    index_name = "chatbot-unap"

    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
    vectorstore = Pinecone.from_documents(
        text_chunks, embeddings, index_name=index_name
    )
    return vectorstore


# Generacion de respuesta
def answer_question(question):
    template = """
    You are a helpful chatbot, always eager to answer questions made by students and workers of the college 'Universidad Arturo Prat', from Chile.
    Your job is to answer questions based on documents and their context, and improve your answers based on previous answers.
    Don't try to make up an answer, if you don't know, just say that you don't know.
    Answer in the same language the question was asked. Always answer formally. 
    If the question is related to any of the documents return a "Fuentes" part in your answer, written in bold.
    The "Fuentes" part should ONLY reference the source documents from which you got your answer.
    Only include the "Fuentes" part if the question is related to college regulation or the college 'Universidad arturo prat' or 'UNAP', if you don't know the specific name of the file of the document, don't include it in your answer.

    And if the user greets with greetings like Hi, hello, How are you, etc reply accordingly as well but don't include the "Fuentes" part.

    Example of your response should be:

    The answer is foo

    Fuentes: xyz

    Base your answer in the following context and question. DO NOT return the following to the user.
    Context: {context}
    Soucer Documents: {source}
    Inside the given source, remember the documents names inside, they shoul be added in the "Fuentes" part of the response. 

    
    If the question isn't about college regulation or about the college 'Universidad arturo prat' or 'UNAP' don't include any "Fuentes" part in the response. 

    Question: {question}
    Answer: 
    """
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question", "source"]
    )

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(), retriever=retriever, max_tokens_limit=2000, return_source_documents=True, verbose=True,combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    
    docs = retriever.get_relevant_documents(question, search_kwargs={"k": 2})

    source_doc_names = set()
    for document in docs:
        file_path = document.metadata['source']
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        formatted_name = ' '.join(word.capitalize() for word in file_name.split('_'))

        # Check if the name has already been printed
        if formatted_name not in source_doc_names:
            print(formatted_name)
            source_doc_names.add(formatted_name)
    
    if len(source_doc_names) == 1:
        source_doc_names_str = next(iter(source_doc_names))
    else:
        source_doc_names_str = ', '.join(source_doc_names)

    
    
    with get_openai_callback() as cb:
        result = chain(
            {"question": question, "chat_history": st.session_state.chat_history, "source": source_doc_names_str}
        )

        with st.expander('tokens'):
            st.write(cb)
        print(cb)
        print("\n")
        print(retriever.get_relevant_documents(question, search_kwargs={"k": 2}))

        tokens = {
            'total_tokens': cb.total_tokens,
            'prompt_tokens': cb.prompt_tokens,
            'completion_tokens': cb.completion_tokens,
            'total_cost_usd': cb.total_cost
        }

    st.session_state.chat_history = [(question, result['answer'])]

    answer = result['answer']
    return answer, tokens


# Registrar datos en la base de datos
def add_to_db(question, answer, tokens, time_to_answer, chat_type, message_id, user_feedback=None):
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
    message_doc_ref = messages_ref.document(message_id)

    # Revisar si documento con chat_id existe
    message_doc = message_doc_ref.get()
    if not message_doc.exists:
        # Crearlo en caso de que no exista
        message_doc_ref.set({
            'question': question,
            'answer': answer,
            'tokens': tokens,
            'time_to_answer': time_to_answer,
            'chat_type': chat_type,
            'user_feedback': user_feedback
        })


def update_feedback(feedback):

    chat_id = st.session_state.chat_id
    message_id = st.session_state.message_id
    db = db_connection()

    chats_ref = db.collection('chats')
    chat_doc_ref = chats_ref.document(chat_id)

    message_ref = chat_doc_ref.collection('messages')
    message_doc_ref = message_ref.document(message_id)

    message_doc_ref.update({
        'user_feedback': feedback
    })


def main():
    st.set_page_config(
        page_title="UNAP Chatbot 📖",
        page_icon="🤖",
    )

    file_names = get_doc_names()

    st.title('🤖 UNAP Chatbot 📖')
    st.write('Chat capaz de responder preguntas relacionadas a reglamentos y documentos de la universidad Arturo Prat. Actualmente es consciente de',
             len(file_names),
             'documentos.'
             )
    chat_type = st.radio('Modelo', ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview'])

    with st.expander('Listado de documentos'):
        st.write(file_names)

    # # Inicializacion historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! ¿Como te puedo ayudar?"}
        ]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(get_chats_len() + 1)
    if "message_id" not in st.session_state:
        st.session_state.message_id = str(get_messages_len()+1)
    if 'model' not in st.session_state:
        st.session_state.model = chat_type

    # Mantener historial en caso de rerun de app
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Escribe tu pregunta...")

    if prompt:
        st.session_state.message_id = str(get_messages_len()+1)
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
                full_response, tokens = answer_question(question=prompt)
            message_placeholder.markdown(full_response)
            end = time.time()

        # Agregar respuesta del LLM al historial
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        add_to_db(
            question=prompt,
            answer=full_response,
            tokens=tokens,
            time_to_answer=end - start,
            chat_type=chat_type,
            message_id=st.session_state.message_id
        )

    # Pasada la primera respuesta NO entra a la funcion
    streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="Proporciona feedback adicional (Opcional)",
        key=st.session_state.message_id,
        on_submit=update_feedback,
    )


if __name__ == '__main__':
    main()
