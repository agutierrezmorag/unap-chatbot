import pinecone
import streamlit as st
from langchain_community.vectorstores import Pinecone as pcvs
from langchain_openai import OpenAIEmbeddings
from termcolor import cprint

from utils import config


def get_pinecone():
    """
    Obtiene una instancia del objeto Pinecone.

    Returns:
        Pinecone: Una instancia del objeto Pinecone.
    """
    return pinecone.Pinecone(
        api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENV
    )


def get_vectorstore(namespace="Reglamentos"):
    """
    Recupera el almacenamiento de vectores para un espacio de nombres dado.

    Args:
        namespace (str): El espacio de nombres del almacenamiento de vectores. Por defecto es "Reglamentos".

    Returns:
        vectorstore (pcvs.VectorStore): El objeto de almacenamiento de vectores.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    vectorstore = pcvs.from_existing_index(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
    )
    return vectorstore


def delete_namespace(namespace):
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
        st.error(f"Hubo un error al eliminar el namespace {namespace}: {e}")


def check_if_index_exists():
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


def get_index_stats():
    """
    Recupera las estadísticas del índice Pinecone.

    Returns:
        dict: Un diccionario que contiene las estadísticas del índice.
    """
    pc = get_pinecone()
    host = pc.describe_index(config.PINECONE_INDEX_NAME).host
    index = pc.Index(host=host)
    index_stats = index.describe_index_stats()

    return index_stats


def delete_all_namespaces():
    """
    Elimina todos los espacios de nombres del índice Pinecone.

    Returns:
        None
    """
    index_stats = get_index_stats()
    for namespace in index_stats.namespaces:
        delete_namespace(namespace)
    cprint("Todos los namespaces eliminados.", "yellow")
    st.success("La memoria de la IA ha sido limpiada.", icon="✅")
