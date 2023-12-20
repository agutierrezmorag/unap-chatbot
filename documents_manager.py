import os
import time
import uuid

import pinecone
import streamlit as st
from github import Auth, Github, GithubException
from github.GithubObject import NotSet
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from st_pages import show_pages_from_config

import streamlit_authenticator as stauth
from register import sign_up, fetch_users

# API keys
OPENAI_API_KEY = st.secrets.openai.api_key
PINECONE_API_KEY = st.secrets.pinecone.api_key
PINECONE_ENV = st.secrets.pinecone.env

# GitHub API keys
REPO_OWNER = st.secrets.github.repo_owner
REPO_NAME = st.secrets.github.repo_name
REPO_BRANCH = st.secrets.github.repo_branch
DIRECTORY_PATH = st.secrets.github.directory_path
ACCESS_TOKEN = st.secrets.github.access_token


@st.cache_resource
def get_repo(show_loader=False):
    """
    Retrieves the GitHub repository object based on the provided access token, repository owner, and repository name.

    Returns:
        repo (github.Repository.Repository): The GitHub repository object.
    """
    auth = Auth.Token(ACCESS_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo(REPO_OWNER + "/" + REPO_NAME)
    return repo


def get_repo_documents():
    """
    Retrieves the documents from the repository.

    Returns:
        A list of documents from the repository.
    """
    try:
        repo = get_repo()
        docs = repo.get_contents(DIRECTORY_PATH, ref=REPO_BRANCH)
        return docs
    except GithubException as e:
        print(f"Error: {e}")
        return []


def delete_doc(file_path, commit_message="Delete file via Streamlit"):
    """
    Deletes a file from the repository.

    Args:
        file_path (str): The path of the file to be deleted.
        commit_message (str, optional): The commit message for the deletion. Defaults to "Delete file via Streamlit".

    Returns:
        bool: True if the file was successfully deleted, False otherwise.
    """
    repo = get_repo()
    try:
        doc = repo.get_contents(file_path, ref=REPO_BRANCH)
        resp = repo.delete_file(
            path=doc.path, message=commit_message, sha=doc.sha, branch=REPO_BRANCH
        )
        return "commit" in resp
    except GithubException as e:
        print(f"Error: {e}")
        return False


def add_files_to_repo(file_list, container, commit_message="Add file via Streamlit"):
    """
    Adds files to a repository.

    Args:
        file_list (list): A list of files to be added.
        commit_message (str, optional): The commit message for the file addition. Defaults to "Add file via Streamlit".
    """
    repo = get_repo()

    for uploaded_file in file_list:
        content = uploaded_file.read().decode("utf-8")

        file_path = f"{DIRECTORY_PATH}/{uploaded_file.name}"

        try:
            existing_file = repo.get_contents(file_path, ref=REPO_BRANCH)
            container.warning(
                f"Documento '{uploaded_file.name}' ya existe. Omitiendo...", icon="⚠️"
            )
            time.sleep(2)
            continue
        except GithubException as e:
            if e.status != 404:
                container.error(
                    f"Error al obtener el documento '{uploaded_file.name}': {e}",
                    icon="❌",
                )
                continue

        try:
            repo.create_file(
                path=file_path,
                message=commit_message,
                content=content,
                branch=REPO_BRANCH,
            )
            container.success(
                f"Documento '{uploaded_file.name}' añadido exitosamente.", icon="✅"
            )
            time.sleep(2)
        except GithubException as e:
            container.error(f"Error al añadir el documento '{uploaded_file.name}': {e}")


# Carga y split de textos
def load_and_split_docs():
    """
    Load and split the documents from the 'documentos' directory.

    Returns:
        texts (list): A list of split text documents.
    """
    raw_text_files = []
    for file in os.listdir("documentos"):
        text_path = "./documentos/" + file
        loader = TextLoader(text_path, encoding="UTF-8")
        raw_text_files.extend(loader.load())

    # Split de textos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=128, length_function=len
    )
    texts = text_splitter.split_documents(raw_text_files)
    return texts


# Crear vectorstore
def do_embedding(text_chunks):
    """
    Embeds the given text chunks using OpenAIEmbeddings and stores them in a Pinecone index.

    Args:
        text_chunks (list): List of text chunks to be embedded.

    Returns:
        vectorstore (Pinecone.VectorStore): The embedded text chunks stored in a Pinecone VectorStore.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
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


def main():
    st.set_page_config(page_title='Documentos', page_icon='📖', initial_sidebar_state='collapsed', layout="wide")


    try:
        users = fetch_users()
        emails = []
        usernames = []
        passwords = []

        for user in users:
            emails.append(user['email'])
            usernames.append(user['username'].lower())
            passwords.append(user['password'])

        credentials = {'usernames': {}}
        for index in range(len(emails)):
            credentials['usernames'][usernames[index]] = {'name': emails[index], 'password': passwords[index]}

        Authenticator = stauth.Authenticate(credentials, cookie_name='Streamlit', key='abcdef', cookie_expiry_days=1)

        email, authentication_status, username = Authenticator.login(':white[Login]', 'main')

        info, info1 = st.columns(2)

        #Para querer registrar una cuenta.
        #if not authentication_status:
            #sign_up()

        if username:
            if username in usernames:
                if authentication_status:
                    # let User see app
                    
                    st.title("📖 Listado de documentos")
                    
                    st.sidebar.subheader(f'Bienvenido {username}')
                    Authenticator.logout('Cerrar Sesión', 'sidebar')

                    show_pages_from_config()

                    container_placeholder = st.empty()

                    if "upload_key" not in st.session_state:
                        st.session_state.upload_key = str(uuid.uuid4())

                    repo_contents = get_repo_documents()

                    if repo_contents:
                        for item in repo_contents:
                            document_path = item.path.replace(DIRECTORY_PATH, "").lstrip("/")
                            document_name, _ = os.path.splitext(document_path)

                            col1, col2 = st.columns([0.9, 0.1])
                            with col1:
                                st.write(f"📑 {document_name}")
                            with col2:
                                delete_button = st.button("⛔", key=document_name)

                            if delete_button:
                                if delete_doc(item.path):
                                    st.success(
                                        f"Documento '{document_name}' eliminado exitosamente.",
                                        icon="✅",
                                    )
                                    st.rerun()
                                else:
                                    st.error(f"Hubo un error al intentar eliminar '{document_name}'.")
                    else:
                        st.info("ℹ️ No hay documentos en el repositorio.")

                    uploaded_files = st.file_uploader(
                        "Sube un nuevo documento",
                        type="txt",
                        accept_multiple_files=True,
                        help="Selecciona uno o más archivos de texto. Solo se permiten archivos .txt.",
                        key=st.session_state.upload_key,
                    )

                    if uploaded_files:
                        if st.button("Subir archivos"):
                            if uploaded_files:
                                add_files_to_repo(uploaded_files, container_placeholder)
                                st.session_state.upload_key = str(uuid.uuid4())
                                st.rerun()

                        if st.button("Limpiar"):
                            st.session_state.upload_key = str(uuid.uuid4())
                            st.rerun()


                elif not authentication_status:
                    with info:
                        st.error('Contraseña o Usuario incorrectos.')
                else:
                    with info:
                        st.warning('Porfavor ingrese las credenciales.')
            else:
                with info:
                    st.warning('Usuario no existente.')


    except:
        st.success('Refresca la pagina.')


if __name__ == "__main__":
    main()
