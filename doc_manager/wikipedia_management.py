import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from pinecone_management import check_if_index_exists, delete_namespace, get_vectorstore
from termcolor import cprint


def load_wikipedia_page(url="https://es.wikipedia.org/wiki/Universidad_Arturo_Prat"):
    """
    Realiza el scraping de una página de Wikipedia, transforma los documentos, los limpia, los divide y los añade a un vector store.

    Args:
        url (str): La URL de la página de Wikipedia a hacer scraping. Por defecto es la página de la Universidad Arturo Prat en español.

    Returns:
        bool: Verdadero si los documentos se añadieron con éxito al vector store, Falso en caso contrario.
    """
    # Scrapping de la pagina de wikipedia
    docs = WikipediaLoader(
        query="Universidad Arturo Prat",
        lang="es",
        load_max_docs=1,
        doc_content_chars_max=20000,
    ).load()

    # Split de los documentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    try:
        check_if_index_exists()

        # Para evitar documentos duplicados, se elimina el namespace si es que existe
        delete_namespace("Wikipedia")

        # Recuperar index
        vectorstore = get_vectorstore("Wikipedia")
        # Añadir documentos
        vectorstore.add_documents(documents=splits)
        cprint("Documentos añadidos al namespace 'Wikipedia'", "green")

        st.toast("Contenido añadido exitosamente.", icon="✅")
    except Exception as e:
        cprint(
            f"Hubo un error al intentar añadir el contenido de Wikipedia al vector store: {e}",
            "red",
        )
        st.toast(
            "Hubo un error al intentar añadir el contenido de Wikipedia.", icon="❌"
        )
