import time

import streamlit as st
from github import Auth, Github, GithubException
from langchain_community.document_loaders import GitLoader
from termcolor import cprint

from doc_manager.pinecone_management import split_and_load_documents_to_vectorstore
from utils import config


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


def delete_repo_doc(file_path, commit_message="Eliminar archivo a través de Streamlit"):
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
        cprint(f"Documento '{doc.name}' eliminado exitosamente.", "yellow")
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


def upload_repo_docs(namespace: str = "Reglamentos"):
    """
    Carga documentos de un repositorio Git, los divide en fragmentos y los añade a un almacenamiento de vectores.

    Args:
        namespace (str, opcional): El espacio de nombres en el almacenamiento de vectores para añadir los documentos. Por defecto es "Reglamentos".

    Throws:
        Exception: Si hay un error al cargar los documentos en el almacenamiento de vectores.
    """
    loader = GitLoader(
        clone_url=config.REPO_URL,
        repo_path=config.REPO_DIRECTORY_PATH,
        branch=config.REPO_BRANCH,
        file_filter=lambda x: x.endswith(".txt"),
    )
    docs = loader.load()
    cprint(f"Cargados {len(docs)} documentos desde {config.REPO_URL}", "yellow")

    split_and_load_documents_to_vectorstore(docs=docs, namespace=namespace)
