import base64
import os
import time
from typing import List

import pandas as pd
import streamlit as st
from github import Auth, ContentFile, Github, GithubException, Repository
from langchain_community.document_loaders import GitLoader
from streamlit.delta_generator import DeltaGenerator
from termcolor import cprint

from doc_manager.pinecone_management import split_and_load_documents_to_vectorstore
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
    file_list: List[str],
    subdirectory: str,
    container: DeltaGenerator,
    progress_bar: DeltaGenerator,
) -> None:
    """
    Añade una lista de archivos a un repositorio de GitHub.

    Esta función recorre una lista de archivos, verifica si cada archivo ya existe en el repositorio y, si no existe, lo añade al repositorio.
    La función actualiza una barra de progreso en cada iteración del bucle.

    Args:
        file_list (List[str]): Una lista de archivos para añadir al repositorio.
        container (DeltaGenerator): Un contenedor de Streamlit para mostrar mensajes.
        progress_bar (DeltaGenerator): Un objeto de barra de progreso de Streamlit para mostrar el progreso de la adición de archivos.

    Returns:
        None
    """
    repo = _get_repo()

    progress_bar.progress(0)
    for i, uploaded_file in enumerate(file_list):
        content = uploaded_file.getvalue()
        file_path = f"{config.REPO_DIRECTORY_PATH}/{subdirectory}/{uploaded_file.name}"

        try:
            repo.get_contents(file_path, ref=config.REPO_BRANCH)
            message = f"El documento '{uploaded_file.name}' ya existe. Omitiendo..."
            cprint(message, "yellow")
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
                    cprint(message, "green")
                except GithubException as e:
                    message = f"Error al añadir el documento '{uploaded_file.name}'"
                    cprint(f"{message} : {e}", "red")
            else:
                message = f"Error al obtener el documento '{uploaded_file.name}'"
                cprint(f"{message} : {e}", "red")

        progress_bar.progress((i + 1) / len(file_list), text=message)
    time.sleep(1)
    progress_bar.progress(1.0, text="Todos los documentos añadidos.")
    time.sleep(1)


def get_files_from_repo(directory_path, namespace) -> None:
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
        file_filter=lambda x: x.endswith(f".{directory_path}"),
    )
    docs = loader.load()
    cprint(f"Cargados {len(docs)} documentos desde {config.REPO_URL}", "yellow")

    if namespace == "Reglamentos":
        split_and_load_documents_to_vectorstore(docs=docs, namespace=namespace)
    elif namespace == "Calendarios":
        split_and_load_documents_to_vectorstore(docs=docs, namespace=namespace)


def clone_repo(token: str, repo_name: str, branch: str, local_dir: str) -> None:
    """
    Clona el contenido de una rama de un repositorio de GitHub en un directorio local.

    Args:
        token (str): El token de acceso personal de GitHub.
        repo_name (str): El nombre del repositorio de GitHub.
        branch (str): La rama del repositorio a clonar.
        local_dir (str): El directorio local donde se clonará el repositorio.

    Returns:
        None
    """
    # Crea una instancia de Github con el token de acceso
    g = Github(token)

    # Obtiene el repositorio
    repo = g.get_repo(repo_name)

    # Obtiene los contenidos de la rama especificada
    contents = repo.get_contents("", ref=branch)

    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path, ref=branch))
        else:
            print(f"Descargando {file_content.path}")
            file_data = base64.b64decode(file_content.content)
            file_path = os.path.join(local_dir, file_content.path)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file:
                file.write(file_data)
