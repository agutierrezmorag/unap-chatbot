from langchain_community.document_loaders import WikipediaLoader
from termcolor import cprint

from doc_manager.pinecone_management import (
    split_and_load_documents_to_vectorstore,
)


def upload_wikipedia_page() -> None:
    """
    Realiza el scraping de una página de Wikipedia, transforma los documentos, los limpia, los divide y los añade a un vector store.

    Args:
        url (str): La URL de la página de Wikipedia a hacer scraping. Por defecto es la página de la Universidad Arturo Prat en español.

    Returns:
        bool: Verdadero si los documentos se añadieron con éxito al vector store, Falso en caso contrario.
    """
    # Scrapping de la pagina de wikipedia
    loader = WikipediaLoader(
        query="Universidad Arturo Prat",
        lang="es",
        load_max_docs=1,
        load_all_available_meta=True,
        doc_content_chars_max=20000,
    )
    docs = loader.load()

    cprint(f"Cargados {len(docs)} documentos desde Wikipedia", "yellow")

    split_and_load_documents_to_vectorstore(docs, "Wikipedia")
