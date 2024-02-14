import os
import shutil
import time
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

import pinecone
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    GitLoader,
    TextLoader,
    UnstructuredXMLLoader,
    WikipediaLoader,
)
from langchain_community.vectorstores import Pinecone as pcvs
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from termcolor import cprint

from utils import config


class SingletonPinecone:
    _instance = None

    @staticmethod
    def get_instance():
        if SingletonPinecone._instance is None:
            SingletonPinecone._instance = pinecone.Pinecone(
                api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENV
            )
        return SingletonPinecone._instance


def _get_pinecone() -> pinecone.Pinecone:
    """
    Obtiene una instancia del objeto Pinecone.

    Returns:
        Pinecone: Una instancia del objeto Pinecone.
    """
    return SingletonPinecone.get_instance()


def _get_or_create_vectorstore(namespace: str) -> pcvs:
    """
    Recupera el almacenamiento de vectores para un espacio de nombres dado. Creando un nuevo espacio de nombres si no existe.

    Args:
        namespace (str): El espacio de nombres del almacenamiento de vectores.

    Returns:
        vectorstore (pcvs): El objeto de almacenamiento de vectores.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    vectorstore = pcvs.from_existing_index(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
    )
    return vectorstore


def delete_namespace(namespace: str) -> None:
    """
    Elimina un espacio de nombres del índice Pinecone.

    Args:
        namespace (str): El nombre del espacio de nombres a eliminar.

    Throws:
        Exception: Si hay un error al eliminar el espacio de nombres.

    Returns:
        None
    """
    try:
        pc = _get_pinecone()
        index = pc.Index(config.PINECONE_INDEX_NAME)
        index.delete(delete_all=True, namespace=namespace)
        cprint(f"Namespace {namespace} eliminado exitosamente.", "yellow")
    except Exception as e:
        cprint(
            f"Hubo un error al eliminar el namespace {namespace}: {e}",
            "red",
        )


def _ensure_index_exists() -> None:
    pc = _get_pinecone()
    if config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        try:
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                metric="cosine",
                dimension=1536,
                spec=pinecone.PodSpec(environment=config.PINECONE_ENV),
            )
            cprint(f"Índice creado {config.PINECONE_INDEX_NAME}", "yellow")
        except Exception as e:
            cprint(f"Error al crear el índice {config.PINECONE_INDEX_NAME}: {e}", "red")


def get_index_data() -> Dict:
    """
    Recupera las estadísticas del índice Pinecone.

    Returns:
        dict: Un diccionario que contiene las estadísticas del índice.
    """
    pc = _get_pinecone()
    host = pc.describe_index(config.PINECONE_INDEX_NAME).host
    index = pc.Index(host=host)
    index_data = index.describe_index_stats()

    return index_data


def delete_all_namespaces() -> None:
    """
    Elimina todos los espacios de nombres del índice Pinecone.

    Returns:
        None
    """
    index_data = get_index_data()
    for namespace in index_data.namespaces:
        delete_namespace(namespace)
    cprint("Todos los namespaces eliminados.", "yellow")


def get_article_urls(index_url: str) -> List[str]:
    """
    Extrae las URLs de los artículos de noticias de la página de índice.

    Args:
        index_url (str): La URL de la página de índice de noticias.

    Returns:
        List[str]: Una lista de URLs de los artículos de noticias.
    """
    response = requests.get(index_url)
    soup = BeautifulSoup(response.text, "html.parser")

    article_links = soup.select(".uap-port-tit a")

    article_urls = [urljoin(index_url, link.get("href")) for link in article_links]

    return article_urls


def get_document_loader(
    namespace: str, path: str, index_url: str = None
) -> Optional[Union[DirectoryLoader, WikipediaLoader]]:
    """
    Obtiene el cargador de documentos basado en el namespace especificado.

    Args:
        namespace (str): El namespace del cargador de documentos.
        path (str): La ruta al directorio o archivo.
        index_url (str): The URL of the news index page. Default is None.

    Returns:
        DocumentLoader: El cargador de documentos basado en el namespace especificado.
    """
    if namespace == "Reglamentos":
        return DirectoryLoader(
            path=path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True, "encoding": "utf-8"},
            use_multithreading=True,
            silent_errors=True,
        )
    elif namespace == "Wikipedia":
        return WikipediaLoader(
            query="Universidad Arturo Prat",
            lang="es",
            load_max_docs=1,
            load_all_available_meta=True,
            doc_content_chars_max=20000,
        )
    elif namespace == "Calendarios":
        return DirectoryLoader(
            path=path,
            glob="**/*.pdf",
            loader_cls=UnstructuredXMLLoader,
            loader_kwargs={"mode": "single"},
            use_multithreading=True,
            silent_errors=True,
        )
    else:
        return None


def process_and_load_documents(
    namespace: str, directory_path: str = None, index_url: str = None
) -> None:
    """
    Procesa y carga documentos desde un directorio a un namespace especificado.

    Args:
        directory_path (str): La ruta al directorio que contiene los documentos.
        namespace (str): El namespace donde se cargarán los documentos.
        index_url (str): The URL of the news index page. Default is None.

    Returns:
        None
    """
    if namespace in ["Reglamentos", "Calendarios"]:
        GitLoader(
            clone_url=config.REPO_URL,
            repo_path=config.REPO_DIRECTORY_PATH,
            branch=config.REPO_BRANCH,
        ).load()

    path = f"{config.REPO_DIRECTORY_PATH}/{config.REPO_DIRECTORY_PATH}/{directory_path}"

    try:
        loader = get_document_loader(namespace, path, index_url)
        if loader is None:
            raise ValueError(f"Namespace no existe en el index: {namespace}")
        docs = loader.load()
    except FileNotFoundError:
        cprint(f"Error: No se encontraron documentos en {path}", "red")
        st.error("No hay documentos en el directorio seleccionado.", icon="📁")
        return

    split_and_store_documents(docs, namespace)


def split_and_store_documents(docs: List[Document], namespace: str) -> None:
    """
    Divide y almacena documentos en un namespace dado.

    Args:
        docs (list): Lista de documentos a dividir y almacenar.
        namespace (str): Namespace donde se almacenarán los documentos.
        path (str): Ruta a los documentos.

    Returns:
        None
    """
    if namespace == "Reglamentos":
        for doc in docs:
            file_name_with_ext = os.path.basename(doc.metadata["source"])
            file_name, _ = os.path.splitext(file_name_with_ext)
            doc.metadata["file_name"] = file_name

    time.sleep(1)
    if namespace in ["Reglamentos", "Calendarios"]:
        try:
            shutil.rmtree(config.REPO_DIRECTORY_PATH)
            cprint("Archivos residuales eliminados.", "blue")
        except PermissionError:
            cprint(
                "Error: No se tienen los permisos necesarios para eliminar el directorio.",
                "red",
            )
        except Exception as e:
            cprint(f"Error: {e}", "red")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    _ensure_index_exists()

    index_data = get_index_data()
    if namespace in index_data.namespaces:
        delete_namespace(namespace)

    try:
        vectorstore = _get_or_create_vectorstore(namespace)
        vectorstore.add_documents(documents=split_docs)

        cprint(
            f"{len(split_docs)} vectores añadidos al namespace '{namespace}'", "green"
        )
    except Exception as e:
        cprint(
            f"Hubo un error al intentar añadir el contenido al vector store: {e}",
            "red",
        )
