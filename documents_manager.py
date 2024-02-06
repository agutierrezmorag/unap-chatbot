import os
import time
import uuid

import pandas as pd
import pinecone
import streamlit as st
import streamlit_authenticator as stauth
from bs4 import BeautifulSoup
from github import Auth, Github, GithubException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    GitLoader,
    WikipediaLoader,
)
from langchain_community.vectorstores import Pinecone as pcvs
from langchain_openai import OpenAIEmbeddings
from st_pages import show_pages_from_config
from termcolor import cprint

from register import fetch_users
from utils import config

logo_path = "logos/unap_negativo.png"


@st.cache_resource
def get_repo(show_loader=False):
    """
    Recupera el objeto del repositorio de GitHub basado en el token de acceso proporcionado, el propietario del repositorio y el nombre del repositorio.

    Returns:
        repo (github.Repository.Repository): El objeto del repositorio de GitHub.
    """
    auth = Auth.Token(config.GITHUB_ACCESS_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo(config.REPO_OWNER + "/" + config.REPO_NAME)

    cprint(f"Repositorio recuperado {config.REPO_OWNER}/{config.REPO_NAME}", "green")

    return repo


@st.cache_resource(
    ttl=60 * 60 * 24, show_spinner="Recuperando listado de documentos..."
)
def get_repo_documents():
    """
    Recupera los documentos del repositorio.

    Returns:
        Una lista de documentos del repositorio.
    """

    try:
        repo = get_repo()
        docs = repo.get_contents(config.REPO_DIRECTORY_PATH, ref=config.REPO_BRANCH)
        cprint(f"Recuperados {len(docs)} documentos del repositorio", "green")
        return docs
    except GithubException as e:
        cprint(f"Error al recuperar documentos del repositorio: {e}", "red")
        return []


def delete_doc(file_path, commit_message="Eliminar archivo a través de Streamlit"):
    """
    Elimina un archivo del repositorio.

    Args:
        file_path (str): La ruta del archivo a eliminar.
        commit_message (str, optional): El mensaje de commit para la eliminación. Por defecto es "Eliminar archivo a través de Streamlit".

    Returns:
        bool: Verdadero si el archivo se eliminó con éxito, Falso en caso contrario.
    """
    repo = get_repo()
    try:
        doc = repo.get_contents(file_path, ref=config.REPO_BRANCH)
        message = f"Documento '{doc.name}' eliminado."
        resp = repo.delete_file(
            path=doc.path,
            message=message,
            sha=doc.sha,
            branch=config.REPO_BRANCH,
        )
        cprint(f"Documento '{doc.name}' eliminado exitosamente.", "green")
        return "commit" in resp
    except GithubException as e:
        cprint(f"Error al eliminar el documento '{file_path}': {e}", "red")
        return False


def add_files_to_repo(file_list, container):
    """
    Añade archivos a un repositorio.

    Args:
        file_list (list): Una lista de archivos a añadir.
        container (streamlit.delta_generator.DeltaGenerator): Un contenedor de Streamlit para mostrar mensajes.
    """
    repo = get_repo()

    for uploaded_file in file_list:
        content = uploaded_file.read().decode("utf-8")

        file_path = f"{config.REPO_DIRECTORY_PATH}/{uploaded_file.name}"

        try:
            existing_file = repo.get_contents(file_path, ref=config.REPO_BRANCH)  # noqa: F841
            container.warning(
                f"El documento '{uploaded_file.name}' ya existe. Omitiendo...", icon="⚠️"
            )
            cprint(
                f"El documento '{uploaded_file.name}' ya existe. Omitiendo...", "yellow"
            )
            time.sleep(2)
            continue
        except GithubException as e:
            if e.status != 404:
                container.error(
                    f"Error al obtener el documento '{uploaded_file.name}': {e}",
                    icon="❌",
                )
                cprint(
                    f"Error al obtener el documento '{uploaded_file.name}': {e}",
                    "red",
                )
                continue

        try:
            message = f"Documento '{uploaded_file.name}' añadido."
            repo.create_file(
                path=file_path,
                message=message,
                content=content,
                branch=config.REPO_BRANCH,
            )
            container.success(
                f"Documento '{uploaded_file.name}' añadido exitosamente.", icon="✅"
            )
            cprint(f"Documento '{uploaded_file.name}' añadido exitosamente.", "green")
            time.sleep(2)
        except GithubException as e:
            container.error(f"Error al añadir el documento '{uploaded_file.name}': {e}")
            cprint(f"Error al añadir el documento '{uploaded_file.name}': {e}", "red")


# Carga y split de textos
def load_and_split_docs():
    """
    Carga y divide los documentos del directorio 'documentos'.

    Returns:
        texts (list): Una lista de documentos de texto divididos.
    """
    loader = GitLoader(
        clone_url=config.REPO_URL,
        repo_path="./documentos",
        branch=config.REPO_BRANCH,
        file_filter=lambda x: x.endswith(".txt"),
    )
    docs = loader.load()
    cprint(f"Cargados {len(docs)} documentos desde {config.REPO_URL}", "yellow")

    # División de textos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=128, length_function=len
    )
    texts = text_splitter.split_documents(docs)
    cprint(f"Documentos divididos en {len(texts)} fragmentos", "green")

    return texts


def scrape_wikipedia_page(url="https://es.wikipedia.org/wiki/Universidad_Arturo_Prat"):
    """
    Realiza el scraping de una página de Wikipedia, transforma los documentos, los limpia, los divide y los añade a un vector store.

    Args:
        url (str): La URL de la página de Wikipedia a hacer scraping. Por defecto es la página de la Universidad Arturo Prat en español.

    Returns:
        bool: Verdadero si los documentos se añadieron con éxito al vector store, Falso en caso contrario.
    """
    # Scrapping de la pagina de wikipedia
    docs = WikipediaLoader(
        query="Universidad Arturo Prat",
        lang="es",
        load_max_docs=1,
        doc_content_chars_max=20000,
    ).load()

    # Split de los documentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
        pc = pinecone.Pinecone(
            api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENV
        )

        if config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                metric="cosine",
                dimension=1536,
                spec=pinecone.PodSpec(environment=config.PINECONE_ENV),
            )

        # Para evitar documentos duplicados, se elimina el namespace si es que existe
        try:
            index = pc.Index(config.PINECONE_INDEX_NAME)
            index.delete(delete_all=True, namespace="Wikipedia")
            cprint("Namespace 'Wikipedia' eliminado", "yellow")
        except pinecone.exceptions.IndexNotFoundError:
            pass

        # Recuperar index
        vectorstore = pcvs.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace="Wikipedia",
        )
        cprint("Namespace 'Wikipedia' creado", "yellow")

        # Añadir documentos
        vectorstore.add_documents(documents=splits)
        cprint("Documentos añadidos al namespace 'Wikipedia'", "green")

        st.success("Documentos añadidos exitosamente.", icon="✅")
    except Exception as e:
        cprint(
            f"Hubo un error al intentar añadir los documentos de Wikipedia al vector store: {e}",
            "red",
        )
        st.error(
            "Hubo un error al intentar añadir los documentos de Wikipedia.", icon="❌"
        )


def clean_up_document(document):
    """
    Limpia el contenido de la página de un documento eliminando las etiquetas HTML y los caracteres especiales.

    Args:
        document (Document): El objeto de documento a limpiar.

    Returns:
        Document: El objeto de documento limpiado.
    """
    cprint("Limpiando el documento: %s", document.metadata["title"], "yellow")

    try:
        soup = BeautifulSoup(document.page_content, "html.parser")
        cleaned_text = (
            soup.get_text()
            .replace("\n", " ")
            .replace("\xa0", " ")
            .replace("\u200b", " ")
        )

        document.page_content = cleaned_text
        cprint("Documento limpiado con éxito: %s", document.metadata["title"], "green")

    except Exception as e:
        cprint(
            f"Error al limpiar el documento: {document.metadata['title']}, error: {e}",
            "red",
        )

    return document


def do_embedding(text_chunks):
    """
    Incrusta los fragmentos de texto dados usando OpenAIEmbeddings y los almacena en un índice Pinecone.

    Args:
        text_chunks (list): Lista de fragmentos de texto para incrustar.

    Returns:
        vectorstore (Pinecone.VectorStore): Los fragmentos de texto incrustados almacenados en un VectorStore de Pinecone.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    pc = pinecone.Pinecone(
        api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENV
    )

    if config.PINECONE_INDEX_NAME in pc.list_indexes().names():
        pc.delete_index(config.PINECONE_INDEX_NAME)
        cprint(f"Índice eliminado {config.PINECONE_INDEX_NAME}", "yellow")

    pc.create_index(
        name=config.PINECONE_INDEX_NAME,
        metric="cosine",
        dimension=1536,
        spec=pinecone.PodSpec(environment=config.PINECONE_ENV),
    )
    cprint(f"Índice creado {config.PINECONE_INDEX_NAME}", "yellow")

    vectorstore = pcvs.from_documents(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        documents=text_chunks,
        namespace="Reglamentos",
    )
    cprint(f"Documentos añadidos al índice {config.PINECONE_INDEX_NAME}", "green")

    return vectorstore


def main():
    st.set_page_config(
        page_title="Documentos",
        page_icon="📖",
        initial_sidebar_state="collapsed",
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
        """,
        unsafe_allow_html=True,
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
            "main",
            fields={
                "Form name": "Inicio de sesión",
                "Username": "Usuario",
                "Password": "Contraseña",
                "Login": "Iniciar Sesión",
            },
        )

        # Para querer registrar una cuenta.
        # if not authentication_status:
        # sign_up()

        if not username:
            st.warning("Por favor, ingrese sus credenciales.")
            st.stop()
        elif username not in usernames:
            st.warning("Usuario no existente.")
            st.stop()
        if not authentication_status:
            st.error("Contraseña o usuario incorrectos.")
            st.stop()
        else:
            st.sidebar.subheader(f"Bienvenido {username}")
            Authenticator.logout("Cerrar Sesión", "sidebar")

            st.header("📚 Gestión de documentos", divider=True)
            st.markdown(
                "En esta sección se pueden gestionar los documentos del repositorio. "
                "Es posible ver los documentos presentes en el repositorio, "
                "subir nuevos documentos o eliminar documentos ya existentes."
            )
            st.info(
                "**Importante**: La IA solo sera consciente de que ha habido modificaciones "
                "en los documentos una vez se [registren los cambios](#Registro).",
                icon="📢",
            )

            show_pages_from_config()

            st.header("🗃️ Documentos en el repositorio", divider=True)
            st.markdown(
                "Listado de documentos presentes en el repositorio. Es posible seleccionar uno o más documentos para eliminarlos."
            )

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
                        ":red[¿Seguro que desea eliminar los documentos seleccionados?]",
                    )
                    if action_button.button("Confirmar"):
                        for i, selected in checkbox_states.items():
                            if selected:
                                document_to_delete = documents_df.loc[i, "File Path"]
                                if delete_doc(document_to_delete):
                                    st.success(
                                        f"Documento '{documents_df.loc[i, 'Document Name']}' eliminado exitosamente."
                                    )
                                    get_repo_documents.clear()
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
                        get_repo_documents.clear()
                        st.rerun()

                if st.button("Limpiar"):
                    st.session_state.upload_key = str(uuid.uuid4())
                    st.rerun()

            st.header("🌐 Wikipedia", divider=True)
            st.markdown(
                "El contenido de la página de Wikipedia de la [Universidad Arturo Prat](https://es.wikipedia.org/wiki/Universidad_Arturo_Prat) "
                "está disponible para añadirse a la base de conocimientos de la IA. "
                "Este contenido puede ser útil para responder preguntas generales sobre la universidad o sobre datos que no estén en los reglamentos. "
                "Si se añade, la IA podrá responder preguntas basándose en esta información."
            )
            st.markdown(
                "Cada vez que se realice esta operación, el contenido anterior de la página de Wikipedia se eliminará y se reemplazará automáticamente "
                "por el contenido actual. Se recomienda hacerlo solo si se está seguro de que el contenido es relevante y actualizado."
            )
            if st.button("Wikipedia"):
                scrape_wikipedia_page()
                time.sleep(2)
                st.rerun()

            st.header("💾 Registrar cambios", anchor="Registro", divider="red")
            st.markdown(
                "Cuando se presione el botón `Registrar cambios`, los documentos que se hayan subido se procesan y \
                se integran en la base de conocimientos de la IA. Solo a partir de ese momento, la IA podrá responder \
                preguntas basándose en la información contenida en estos documentos."
            )
            st.info(
                "Este proceso puede tardar varios minutos. No refresque la página mientras se esté realizando el registro.",
                icon="💡",
            )
            if st.button("Registrar cambios", type="primary", use_container_width=True):
                texts = load_and_split_docs()
                if do_embedding(texts):
                    st.success(
                        "✅ Documentos registrados exitosamente. Ahora es posible realizar consultas sobre los nuevos documentos."
                    )
                    time.sleep(5)
                    st.rerun()
                else:
                    st.error("❌ Hubo un error al registrar los documentos.")

    except Exception as e:
        print(e)
        st.rerun()


if __name__ == "__main__":
    main()
