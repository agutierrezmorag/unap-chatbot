import logging
import time
from typing import List

import pandas as pd
import streamlit as st
from github import Auth, ContentFile, Github, GithubException, Repository
from streamlit.elements.widgets.file_uploader import UploadedFile
from termcolor import cprint

from doc_manager.pinecone_management import process_and_load_documents
from utils import config

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _get_repo() -> Repository.Repository:
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
def get_repo_documents(subdirectory: str) -> List[ContentFile]:
    """
    Recupera los documentos del repositorio.

    Returns:
        Una lista de documentos del repositorio.
    """

    try:
        repo = _get_repo()
        directory_path = f"{config.REPO_DIRECTORY_PATH}/{subdirectory}"
        docs = repo.get_contents(directory_path, ref=config.REPO_BRANCH)
        cprint(f"Recuperados {len(docs)} documentos del repositorio", "green")
        return docs
    except GithubException as e:
        cprint(f"Error al recuperar documentos del repositorio: {e}", "red")
        return []


def get_repo_docs_as_pd(subdirectory: str) -> pd.DataFrame:
    """
    Converts a list of ContentFile objects into a pandas DataFrame.

    Args:
        content_files (List[ContentFile]): A list of ContentFile objects.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a ContentFile.
    """
    content_files = get_repo_documents(subdirectory)
    data = []
    for file in content_files:
        data.append(
            {
                "type": file.type,
                "path": file.path,
                "name": file.name,
                "sha": file.sha,
                "html_url": file.html_url,
                "size": file.size / 1024,
                "download_url": file.download_url,
                "selected": False,
            }
        )
    return pd.DataFrame(data)


def delete_repo_doc(
    file_path: str, commit_message: str = "Eliminar archivo a través de Streamlit"
) -> bool:
    """
    Elimina un archivo del repositorio.

    Args:
        file_path (str): La ruta del archivo a eliminar.
        commit_message (str, optional): El mensaje de commit para la eliminación. Por defecto es "Eliminar archivo a través de Streamlit".

    Returns:
        bool: Verdadero si el archivo se eliminó con éxito, Falso en caso contrario.
    """
    repo = _get_repo()
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


def add_files_to_repo(
    file_list: List[UploadedFile],
    subdirectory: str,
    namespace: str,
) -> None:
    """
    Añade archivos a un repositorio de GitHub. Luego, procesa y carga los documentos en Pinecone.

    Args:
        file_list (List[UploadedFile]): Lista de archivos a añadir.
        subdirectory (str): Subdirectorio dentro del repositorio donde añadir los archivos.
        namespace (str): Espacio de nombres de los documentos en Pinecone.
        progress_bar (DeltaGenerator): Objeto de barra de progreso para mostrar el progreso.

    Returns:
        None
    """
    repo = _get_repo()

    update_container = st.status(
        label="Añadiendo documentos al repositorio...",
        state="running",
        expanded=True,
    )
    for uploaded_file in file_list:
        content = uploaded_file.getvalue()
        file_path = f"{config.REPO_DIRECTORY_PATH}/{subdirectory}/{uploaded_file.name}"

        try:
            repo.get_contents(file_path, ref=config.REPO_BRANCH)
            message = f"Documento '{uploaded_file.name}' ya existe. Omitiendo..."
            logger.warning(message)
        except GithubException as e:
            if e.status == 404:
                try:
                    message = f"Documento '{uploaded_file.name}' añadido."
                    repo.create_file(
                        path=file_path,
                        message=message,
                        content=content,
                        branch=config.REPO_BRANCH,
                    )
                    logger.info(message)
                except GithubException as e:
                    message = f"Error al añadir el documento '{uploaded_file.name}'"
                    logger.error(f"{message} : {e}")
            else:
                message = f"Error al obtener el documento '{uploaded_file.name}'"
                logger.error(f"{message} : {e}")

        update_container.markdown(f"- {message}")
    time.sleep(1)
    update_container.markdown("- Documentos cargados exitosamente.")
    time.sleep(1)

    update_container.markdown("- Actualizando la memoria de la IA...")
    process_and_load_documents(namespace=namespace, directory_path=subdirectory)
    update_container.update(label="Memoria actualizada.", state="complete")
