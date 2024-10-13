import logging
from typing import List

import pandas as pd
import streamlit as st
from document_manager.pinecone_management import process_and_load_documents
from github import Auth, ContentFile, Github, GithubException, Repository
from streamlit.elements.widgets.file_uploader import UploadedFile

from utils import config

logging.basicConfig(level=logging.INFO)


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

    logging.info(f"Repositorio recuperado {config.REPO_OWNER}/{config.REPO_NAME}")

    return repo


def _get_repo_documents(subdirectory: str) -> List[ContentFile]:
    """
    Recupera los documentos del repositorio.

    Returns:
        Una lista de documentos del repositorio.
    """

    try:
        repo = _get_repo()
        directory_path = f"{config.REPO_DIRECTORY_PATH}/{subdirectory}"
        docs = repo.get_contents(directory_path, ref=config.REPO_BRANCH)
        logging.debug(
            f"Recuperados {len(docs)} documentos del repositorio, desde {directory_path}."
        )
        return docs
    except GithubException as e:
        logging.error(f"Error al recuperar documentos del repositorio: {e}")
        return []


def get_repo_docs_as_pd(subdirectory: str) -> pd.DataFrame:
    """
    Obtiene los documentos del repositorio en formato pandas DataFrame.

    Args:
        subdirectory (str): Subdirectorio del repositorio.

    Returns:
        pd.DataFrame: DataFrame que contiene la información de los documentos del repositorio.
    """
    content_files = _get_repo_documents(subdirectory)
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


def delete_repo_doc(file_paths: List[str], namespace: str) -> None:
    """
    Elimina los documentos especificados del repositorio y actualiza la memoria de la IA.

    Args:
        file_paths (List[str]): Lista de rutas de los archivos a eliminar.
        namespace (str): Namespace en Pinecone.
        directory_path (str): Ruta del directorio donde se encuentran los documentos.

    Returns:
        None
    """
    repo = _get_repo()
    progress_bar = st.progress(0)
    total_files = len(file_paths)
    try:
        for i, file_path in enumerate(file_paths):
            doc = repo.get_contents(file_path, ref=config.REPO_BRANCH)
            message = f"Documento '{doc.name}' eliminado."
            resp = repo.delete_file(  # noqa: F841
                path=doc.path,
                message=message,
                sha=doc.sha,
                branch=config.REPO_BRANCH,
            )
            progress_bar.progress((i + 1) / total_files, text=message)

        progress_bar.progress(1.0, text=":green[Documentos eliminados exitosamente.]")

        progress_bar.progress(1.0, text=":blue[Actualizando la memoria de la IA...]")
        process_and_load_documents(namespace=namespace)
        progress_bar.progress(1.0, text="Memoria actualizada.")
    except GithubException as e:
        logging.error(f"Error al eliminar el documento '{file_path}': {e}")
        return


def add_files_to_repo(file_list: List[UploadedFile], namespace: str) -> None:
    """
    Agrega archivos a un repositorio de GitHub. Luego, procesa y carga los documentos en Pinecone.

    Args:
        file_list (List[UploadedFile]): Lista de archivos para agregar.
        namespace (str): Namespace de los documentos en Pinecone.

    Returns:
        None
    """
    repo = _get_repo()

    progress_bar = st.progress(0)
    total_files = len(file_list)
    for i, uploaded_file in enumerate(file_list):
        content = uploaded_file.getvalue()
        file_path = f"{config.REPO_DIRECTORY_PATH}/{namespace}/{uploaded_file.name}"

        try:
            repo.get_contents(file_path, ref=config.REPO_BRANCH)
            message = f"El documento '{uploaded_file.name}' ya existe. Saltando..."
            logging.warning(message)
        except GithubException as e:
            if e.status == 404:
                try:
                    message = f"Documento '{uploaded_file.name}' agregado."
                    repo.create_file(
                        path=file_path,
                        message=message,
                        content=content,
                        branch=config.REPO_BRANCH,
                    )
                    logging.info(message)
                except GithubException as e:
                    message = f"Error al agregar el documento '{uploaded_file.name}'"
                    logging.error(f"{message} : {e}")
            else:
                message = f"Error al obtener el documento '{uploaded_file.name}'"
                logging.error(f"{message} : {e}")

        progress_bar.progress(value=(i + 1) / total_files, text=message)
    progress_bar.progress(value=1.0, text=":green[Documentos cargados exitosamente.]")

    progress_bar.progress(value=1.0, text=":blue[Actualizando la memoria de la IA...]")
    process_and_load_documents(namespace=namespace)
    progress_bar.progress(value=1.0, text="Memoria actualizada.")
