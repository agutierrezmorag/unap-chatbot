import logging
import time
from typing import List

import pandas as pd
import streamlit as st
from github import Auth, ContentFile, Github, GithubException, Repository
from streamlit.elements.widgets.file_uploader import UploadedFile

from doc_manager.pinecone_management import process_and_load_documents
from utils import config


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
        logging.info(f"Recuperados {len(docs)} documentos del repositorio")
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


def delete_repo_doc(file_paths: List[str], namespace: str, directory_path: str) -> None:
    """
    Elimina los documentos especificados del repositorio y actualiza la memoria de la IA.

    Args:
        file_paths (List[str]): Lista de rutas de los archivos a eliminar.
        namespace (str): Espacio de nombres en Pinecone.
        directory_path (str): Ruta del directorio donde se encuentran los documentos.

    Returns:
        None
    """
    repo = _get_repo()
    progress_container = st.status(
        "Eliminando documentos...", state="running", expanded=True
    )
    try:
        for file_path in file_paths:
            doc = repo.get_contents(file_path, ref=config.REPO_BRANCH)
            message = f"Documento '{doc.name}' eliminado."
            resp = repo.delete_file(  # noqa: F841
                path=doc.path,
                message=message,
                sha=doc.sha,
                branch=config.REPO_BRANCH,
            )
            progress_container.markdown(f"- {message}")

        progress_container.markdown("- Documentos eliminados exitosamente.")

        progress_container.markdown("- Actualizando la memoria de la IA...")
        process_and_load_documents(namespace=namespace, directory_path=directory_path)
        progress_container.update(label="Memoria actualizada.", state="complete")
        time.sleep(1)
    except GithubException as e:
        logging.error(f"Error al eliminar el documento '{file_path}': {e}")
        return


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
            logging.warning(message)
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
                    logging.info(message)
                except GithubException as e:
                    message = f"Error al añadir el documento '{uploaded_file.name}'"
                    logging.error(f"{message} : {e}")
            else:
                message = f"Error al obtener el documento '{uploaded_file.name}'"
                logging.error(f"{message} : {e}")

        update_container.markdown(f"- {message}")
    time.sleep(1)
    update_container.markdown("- Documentos cargados exitosamente.")
    time.sleep(1)

    update_container.markdown("- Actualizando la memoria de la IA...")
    process_and_load_documents(namespace=namespace, directory_path=subdirectory)
    update_container.update(label="Memoria actualizada.", state="complete")
