from typing import Dict, List

import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as pcvs
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


def get_pinecone():
    """
    Obtiene una instancia del objeto Pinecone.

    Returns:
        Pinecone: Una instancia del objeto Pinecone.
    """
    return SingletonPinecone.get_instance()


def get_or_create_vectorstore(namespace: str) -> pcvs:
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
        pc = get_pinecone()
        index = pc.Index(config.PINECONE_INDEX_NAME)
        index.delete(delete_all=True, namespace=namespace)
        cprint(f"Namespace {namespace} eliminado exitosamente.", "yellow")
    except Exception as e:
        cprint(
            f"Hubo un error al eliminar el namespace {namespace}: {e}",
            "red",
        )


def ensure_index_exists() -> None:
    pc = get_pinecone()
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
    pc = get_pinecone()
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


def split_and_load_documents_to_vectorstore(docs: List[str], namespace: str) -> None:
    """
    Divide documentos y los añade a un almacenamiento de vectores.

    Args:
        docs (list): Una lista de documentos a añadir al almacenamiento de vectores.
        namespace (str): El espacio de nombres en el almacenamiento de vectores para añadir los documentos.

    Throws:
        Exception: Si hay un error al cargar los documentos en el almacenamiento de vectores.

    Returns:
        None
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    ensure_index_exists()

    index_data = get_index_data()
    if namespace in index_data.namespaces:
        delete_namespace(namespace)

    try:
        vectorstore = get_or_create_vectorstore(namespace)
        vectorstore.add_documents(documents=split_docs)

        cprint(
            f"{len(split_docs)} vectores añadidos al namespace '{namespace}'", "green"
        )
    except Exception as e:
        cprint(
            f"Hubo un error al intentar añadir el contenido al vector store: {e}",
            "red",
        )
