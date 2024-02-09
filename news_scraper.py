from doc_manager.pinecone_management import process_and_load_documents
from utils import config

process_and_load_documents(
    "news",
    "Noticias",
    "https://www.unap.cl/prontus_unap/site/edic/base/port/actualidad.html",
)
