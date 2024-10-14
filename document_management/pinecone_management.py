import logging
import os
import shutil
import time
from typing import Dict, List, Optional, Union

import pinecone
import streamlit as st
from langchain_community.document_loaders import (
    DirectoryLoader,
    GitLoader,
    TextLoader,
    UnstructuredXMLLoader,
    WikipediaLoader,
)
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import config


class SingletonPinecone:
    _instance = None

    @staticmethod
    def get_instance():
        if SingletonPinecone._instance is None:
            SingletonPinecone._instance = pinecone.Pinecone(
                api_key=config.PINECONE_API_KEY
            )
        return SingletonPinecone._instance


def _get_pinecone_instance() -> pinecone.Pinecone:
    """
    Obtiene una instancia del objeto Pinecone.

    Returns:
        Pinecone: Una instancia del objeto Pinecone.
    """
    return SingletonPinecone.get_instance()


def get_embedder() -> OpenAIEmbeddings:
    """
    Obtiene el objeto OpenAIEmbeddings para realizar la incrustación de texto.

    Returns:
        OpenAIEmbeddings: El objeto OpenAIEmbeddings para realizar la incrustación de texto.
    """
    return OpenAIEmbeddings(
        openai_api_key=config.OPENAI_API_KEY, model="text-embedding-3-small"
    )


def get_or_create_vectorstore(namespace: str) -> Pinecone:
    """
    Recupera el almacenamiento de vectores para un namespace dado. Creando un nuevo namespace si no existe.

    Args:
        namespace (str): El namespace en Pinecone.

    Returns:
        vectorstore (Pinecone): El objeto de almacenamiento de vectores.
    """
    embeddings = get_embedder()
    vectorstore = Pinecone.from_existing_index(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
    )
    return vectorstore


def delete_namespace(namespace: str) -> None:
    """
    Elimina un namespace del índice Pinecone.

    Args:
        namespace (str): El nombre del namespace a eliminar.

    Throws:
        Exception: Si hay un error al eliminar el namespace.

    Returns:
        None
    """
    try:
        pc = _get_pinecone_instance()
        index = pc.Index(config.PINECONE_INDEX_NAME)
        index.delete(delete_all=True, namespace=namespace)
        logging.info(f"Namespace {namespace} eliminado exitosamente.")
    except Exception as e:
        if "Namespace not found" in str(e):
            logging.info(f"El namespace {namespace} no existe. No se eliminó nada.")
            pass
        else:
            logging.error(f"Hubo un error al eliminar el namespace {namespace}: {e}")


def _ensure_index_exists() -> None:
    """
    Asegura que el índice exista en Pinecone.

    Verifica si el índice especificado en la configuración existe en Pinecone.
    Si no existe, crea el índice con las especificaciones dadas.

    Raises:
        Exception: Si ocurre un error al crear el índice.

    Returns:
        None
    """
    pc = _get_pinecone_instance()
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
    pc = _get_pinecone_instance()
    host = pc.describe_index(config.PINECONE_INDEX_NAME).host
    index = pc.Index(host=host)
    index_data = index.describe_index_stats()

    return index_data


@st.cache_resource(show_spinner=False)
def _get_document_loader(
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
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
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
            loader_kwargs={"mode": "single", "strategy": "hi_res"},
            use_multithreading=True,
            silent_errors=True,
        )
    elif namespace == "Web":
        return DirectoryLoader(
            path=path,
            glob="**/*.xml",
            loader_cls=SitemapLoader,
            loader_kwargs={"is_local": True, "continue_on_failure": True},
            use_multithreading=True,
            silent_errors=True,
        )
    else:
        return None


def process_and_load_documents(namespace: str) -> None:
    """
    Procesa y carga documentos desde un directorio a un namespace especificado.

    Args:
        namespace (str): El namespace donde se cargarán los documentos.

    Returns:
        None
    """
    if namespace != "Wikipedia":
        GitLoader(
            clone_url=config.REPO_URL,
            repo_path=config.REPO_DIRECTORY_PATH,
            branch=config.REPO_BRANCH,
        ).load()

    path = f"{config.REPO_DIRECTORY_PATH}/{config.REPO_DIRECTORY_PATH}/{namespace}"

    try:
        loader = _get_document_loader(namespace, path)
        if loader is None:
            raise ValueError(f"Namespace no existe en el index: {namespace}")
        docs = loader.load()
    except FileNotFoundError:
        delete_namespace(namespace)
        logging.error(f"Error: No se encontraron documentos en {path}")
        return

    _split_and_store_documents(docs, namespace)


def _split_and_store_documents(docs: List[Document], namespace: str) -> None:
    """
    Divide y almacena documentos en un namespace dado.

    Args:
        docs (list): Lista de documentos a dividir y almacenar.
        namespace (str): Namespace donde se almacenarán los documentos.
        path (str): Ruta a los documentos.

    Returns:
        None
    """
    cloned_namespaces = ["Reglamentos", "Calendarios"]
    if namespace in cloned_namespaces:
        for doc in docs:
            normalized_path = doc.metadata["source"].replace("\\", "/")
            file_name_with_ext = os.path.basename(normalized_path)
            file_name, _ = os.path.splitext(file_name_with_ext)
            doc.metadata["file_name"] = file_name

    time.sleep(1)
    if namespace in cloned_namespaces:
        try:
            shutil.rmtree(config.REPO_DIRECTORY_PATH)
            logging.info("Archivos residuales eliminados.")
        except PermissionError:
            logging.critical(
                f"Error: No se tienen los permisos necesarios para eliminar el directorio '{config.REPO_DIRECTORY_PATH}. Por favor, elimine el directorio manualmente."
            )
        except Exception as e:
            logging.error(f"Error: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=300, add_start_index=True
    )
    split_docs = splitter.split_documents(docs)

    _ensure_index_exists()

    index_data = get_index_data()
    if namespace in index_data.namespaces:
        delete_namespace(namespace)

    try:
        vectorstore = get_or_create_vectorstore(namespace)
        vectorstore.add_documents(documents=split_docs)

        logging.info(f"{len(split_docs)} vectores añadidos al namespace '{namespace}'")
    except Exception as e:
        logging.error(
            f"Hubo un error al intentar añadir el contenido al vector store: {e}"
        )
