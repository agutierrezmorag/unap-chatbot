import logging
import os
import shutil
import time
from typing import Dict, List, Optional, Union

import pinecone
import streamlit as st
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

from utils import config

logging.basicConfig(level=logging.INFO)


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
        logging.info(f"Namespace {namespace} eliminado exitosamente.")
    except Exception as e:
        logging.error(f"Hubo un error al eliminar el namespace {namespace}: {e}")


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
            logging.info(f"Índice creado {config.PINECONE_INDEX_NAME}")
        except Exception as e:
            logging.error(f"Error al crear el índice {config.PINECONE_INDEX_NAME}: {e}")


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
    logging.info("Todos los namespaces eliminados.")


def get_document_loader(
    namespace: str, path: str
) -> Optional[Union[DirectoryLoader, WikipediaLoader]]:
    """
    Obtiene el cargador de documentos basado en el namespace especificado.

    Args:
        namespace (str): El namespace del cargador de documentos.
        path (str): La ruta al directorio o archivo.

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
            glob="**/*.xml",
            loader_cls=UnstructuredXMLLoader,
            loader_kwargs={"mode": "single"},
            use_multithreading=True,
            silent_errors=True,
        )
    else:
        return None


def process_and_load_documents(namespace: str, directory_path: str = None) -> None:
    """
    Procesa y carga documentos desde un directorio a un namespace especificado.

    Args:
        directory_path (str): La ruta al directorio que contiene los documentos.
        namespace (str): El namespace donde se cargarán los documentos.

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
        loader = get_document_loader(namespace, path)
        if loader is None:
            raise ValueError(f"Namespace no existe en el index: {namespace}")
        docs = loader.load()
    except FileNotFoundError:
        logging.error(f"Error: No se encontraron documentos en {path}")
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
    if namespace in ["Reglamentos", "Calendarios"]:
        for doc in docs:
            normalized_path = doc.metadata["source"].replace("\\", "/")
            file_name_with_ext = os.path.basename(normalized_path)
            file_name, _ = os.path.splitext(file_name_with_ext)
            doc.metadata["file_name"] = file_name

    time.sleep(1)
    if namespace in ["Reglamentos", "Calendarios"]:
        try:
            shutil.rmtree(config.REPO_DIRECTORY_PATH)
            logging.info("Archivos residuales eliminados.")
        except PermissionError:
            logging.error(
                "Error: No se tienen los permisos necesarios para eliminar el directorio."
            )
        except Exception as e:
            logging.error(f"Error: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    _ensure_index_exists()

    index_data = get_index_data()
    if namespace in index_data.namespaces:
        delete_namespace(namespace)

    try:
        vectorstore = _get_or_create_vectorstore(namespace)
        vectorstore.add_documents(documents=split_docs)

        logging.info(f"{len(split_docs)} vectores añadidos al namespace '{namespace}'")
    except Exception as e:
        logging.error(
            f"Hubo un error al intentar añadir el contenido al vector store: {e}"
        )
