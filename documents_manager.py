import os
import time
import uuid

import pandas as pd
import pinecone
import streamlit as st
import streamlit_authenticator as stauth
from github import Auth, Github, GithubException
from langchain.document_loaders import GitLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from st_pages import show_pages_from_config

from register import fetch_users
from utils import config

logo_path = "logos/unap_negativo.png"

@st.cache_resource
def get_repo(show_loader=False):
    """
    Retrieves the GitHub repository object based on the provided access token, repository owner, and repository name.

    Returns:
        repo (github.Repository.Repository): The GitHub repository object.
    """
    auth = Auth.Token(config.GITHUB_ACCESS_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo(config.REPO_OWNER + "/" + config.REPO_NAME)
    return repo


def get_repo_documents():
    """
    Retrieves the documents from the repository.

    Returns:
        A list of documents from the repository.
    """
    try:
        repo = get_repo()
        docs = repo.get_contents(config.REPO_DIRECTORY_PATH, ref=config.REPO_BRANCH)
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
        doc = repo.get_contents(file_path, ref=config.REPO_BRANCH)
        resp = repo.delete_file(
            path=doc.path,
            message=commit_message,
            sha=doc.sha,
            branch=config.REPO_BRANCH,
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

        file_path = f"{config.REPO_DIRECTORY_PATH}/{uploaded_file.name}"

        try:
            existing_file = repo.get_contents(file_path, ref=config.REPO_BRANCH)
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
                branch=config.REPO_BRANCH,
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

    loader = GitLoader(
        clone_url=config.REPO_URL,
        repo_path="./documentos",
        branch=config.REPO_BRANCH,
        file_filter=lambda x: x.endswith(".txt"),
    )
    docs = loader.load()

    # Split de textos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=128, length_function=len
    )
    texts = text_splitter.split_documents(docs)

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
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    pinecone.init(
        api_key=config.PINECONE_API_KEY,
        environment=config.PINECONE_ENV,
    )

    # Revisar si el index ya existe
    if config.PINECONE_INDEX_NAME in pinecone.list_indexes():
        # Borrarlo en caso de que exista
        pinecone.delete_index(config.PINECONE_INDEX_NAME)

    # Crear un nuevo index
    pinecone.create_index(
        name=config.PINECONE_INDEX_NAME, metric="cosine", dimension=1536
    )
    vectorstore = Pinecone.from_documents(
        text_chunks, embeddings, index_name=config.PINECONE_INDEX_NAME
    )
    return vectorstore


def main():
    st.set_page_config(
        page_title="Documentos",
        page_icon="📖",
    )

    st.markdown(
        """
        <style>
            [data-testid=stSidebar] [data-testid=stImage]{
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
                margin-top: auto;
                width: 100%;
            }
        </style>
        """, unsafe_allow_html=True
    )

    with st.sidebar:
        st.image(logo_path, use_column_width=True)

    if "upload_key" not in st.session_state:
        st.session_state.upload_key = str(uuid.uuid4())
    if "delete_selected" not in st.session_state:
        st.session_state.delete_selected = False

    try:
        users = fetch_users()
        emails = []
        usernames = []
        passwords = []

        for user in users:
            emails.append(user["email"])
            usernames.append(user["username"].lower())
            passwords.append(user["password"])

        credentials = {"usernames": {}}
        for index in range(len(emails)):
            credentials["usernames"][usernames[index]] = {
                "name": emails[index],
                "password": passwords[index],
            }

        Authenticator = stauth.Authenticate(
            credentials, cookie_name="Streamlit", key="abcdef", cookie_expiry_days=1
        )

        email, authentication_status, username = Authenticator.login(
            ":white[Login]", "main"
        )

        # Para querer registrar una cuenta.
        # if not authentication_status:
        # sign_up()

        if username:
            if username in usernames:
                if authentication_status:
                    # let User see app
                    st.sidebar.subheader(f"Bienvenido {username}")
                    Authenticator.logout("Cerrar Sesión", "sidebar")

                    st.title("⚙️ Gestión de documentos")
                    with st.expander("ℹ️ Información"):
                        st.markdown(
                            """
                            ## 📋 Instrucciones
                            ### Cargar nuevos documentos
                            - Se debe subir uno o mas archivos de texto (.txt).
                            - Presionar el boton 'Subir archivos'.
                            - Se dispone de un boton 'Limpiar' para limpiar la lista de archivos subidos.

                            ### Eliminar documentos
                            - Seleccionar el/los documentos a eliminar.
                            - Presionar el boton 'Eliminar documentos seleccionados'.
                            - Confirmar la eliminacion de los documentos.
                            """
                        )

                    show_pages_from_config()

                    st.markdown("## 📂 Documentos en el repositorio")

                    container_placeholder = st.empty()

                    repo_contents = get_repo_documents()

                    if repo_contents:
                        # Create a list to store the documents data
                        documents_data = []

                        for item in repo_contents:
                            document_path = item.path.replace(
                                config.REPO_DIRECTORY_PATH, ""
                            ).lstrip("/")
                            document_name, _ = os.path.splitext(document_path)

                            # Append a dictionary with the document data to the list
                            documents_data.append(
                                {
                                    "Document Name": document_name,
                                    "File Path": item.path,
                                    "Selected": False,
                                }
                            )

                        # Create a DataFrame from the documents data
                        documents_df = pd.DataFrame(documents_data)

                        # Create a dictionary to store the checkbox states
                        checkbox_states = {}

                        # Display the DataFrame with checkboxes
                        with st.container(border=True):
                            for i in range(len(documents_df)):
                                checkbox_states[i] = st.checkbox(
                                    documents_df.loc[i, "Document Name"], key=i
                                )

                        # Create placeholders for the buttons
                        confirm_dialog = st.empty()
                        action_button = st.empty()
                        cancel_button = st.empty()

                        # Display the appropriate action button
                        if st.session_state.get("delete_selected"):
                            confirm_dialog.markdown(
                                ":red[¿Estás seguro de que deseas eliminar los documentos seleccionados?]",
                            )
                            if action_button.button("Confirmar"):
                                for i, selected in checkbox_states.items():
                                    if selected:
                                        document_to_delete = documents_df.loc[
                                            i, "File Path"
                                        ]
                                        if delete_doc(document_to_delete):
                                            st.success(
                                                f"Documento '{documents_df.loc[i, 'Document Name']}' eliminado exitosamente."
                                            )
                                        else:
                                            st.error(
                                                f"Hubo un error al intentar eliminar '{documents_df.loc[i, 'Document Name']}'."
                                            )
                                st.session_state.delete_selected = False
                                time.sleep(2)
                                st.rerun()
                            elif cancel_button.button("Cancelar"):
                                st.session_state.delete_selected = False
                                st.rerun()
                        else:
                            action_button = st.button(
                                "Eliminar documentos seleccionados",
                                disabled=not any(checkbox_states.values()),
                            )
                            if action_button:
                                st.session_state.delete_selected = True
                                st.rerun()
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

                    st.divider()

                    st.markdown("## 💾 Registrar cambios")

                    st.markdown(
                        "Cuando se presione el botón `Registrar cambios`, los documentos que se hayan subido se procesan y \
                        se integran en la base de conocimientos de la IA. A partir de ese momento, la IA podrá responder \
                        preguntas basándose en la información contenida en estos documentos."
                    )

                    st.info(
                        "**Es importante recordar que el procesamiento puede llevar algún tiempo, dependiendo del tamaño y la cantidad de los documentos subidos.**",
                        icon="📢",
                    )

                    if st.columns(3)[1].button("Registrar cambios"):
                        texts = load_and_split_docs()
                        if do_embedding(texts):
                            st.success("✅ Documentos registrados exitosamente.")
                            st.rerun()
                        else:
                            st.error("❌ Hubo un error al registrar los documentos.")

                elif not authentication_status:
                    st.error("Contraseña o Usuario incorrectos.")
                else:
                    st.warning("Por favor, ingrese sus credenciales.")
            else:
                st.warning("Usuario no existente.")

    except stauth.StreamlitAuthenticationException as e:
        print(e)
        st.rerun()


if __name__ == "__main__":
    main()
