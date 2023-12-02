import os

import pinecone
import streamlit as st
from dotenv import load_dotenv
from github import Auth, Github
from github.GithubObject import NotSet
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# GitHub API keys
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")
DIRECTORY_PATH = os.getenv("DIRECTORY_PATH")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")


def get_repo():
    auth = Auth.Token(ACCESS_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo(REPO_OWNER + "/" + REPO_NAME)
    return repo


def get_repo_documents():
    repo = get_repo()
    docs = repo.get_contents("documentos")
    return docs


def delete_doc(file_path, commit_message="Delete file via Streamlit"):
    repo = get_repo()
    doc = repo.get_contents(file_path)
    resp = repo.delete_file(
        path=doc.path, message=commit_message, sha=doc.sha, branch="test"
    )
    print(resp)
    return resp["content"] == NotSet


def add_files_to_repo(file_list, commit_message="Add file via Streamlit"):
    repo = get_repo()

    for uploaded_file in file_list:
        content = uploaded_file.read()

        # Specify the path to save the file in the repository
        file_path = f"{DIRECTORY_PATH}/{uploaded_file.name}"

        # Check if the file already exists in the repository
        try:
            existing_file = repo.get_contents(file_path)
            st.warning(f"Documento '{uploaded_file.name}' ya existe. Omitiendo...")
            continue
        except Exception as e:
            # File doesn't exist, proceed with creating it
            pass

        # Create the new file in the repository
        repo.create_file(
            path=file_path,
            message=commit_message,
            content=content,
            branch="test",
        )

        st.success(f"Documento '{uploaded_file.name}' añadido exitosamente.", icon="✅")


# Carga y split de textos
def load_and_split_docs():
    # Carga de documentos
    raw_text_files = []
    for file in os.listdir("documentos"):
        text_path = "./documentos/" + file
        loader = TextLoader(text_path, encoding="UTF-8")
        raw_text_files.extend(loader.load())

    # Split de textos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=128, length_function=len
    )
    texts = text_splitter.split_documents(raw_text_files)
    return texts


# Crear vectorstore
def do_embedding(text_chunks):
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
    )
    index_name = "chatbot-unap"

    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
    vectorstore = Pinecone.from_documents(
        text_chunks, embeddings, index_name=index_name
    )
    return vectorstore


def main():
    st.title("Listado de documentos")

    repo_contents = get_repo_documents()

    if repo_contents:
        for item in repo_contents:
            _, document_name = item.path.split("documentos/")
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"📑 {document_name}")
            with col2:
                delete_button = st.button("⛔", key=document_name)

            if delete_button:
                # Ask for confirmation before deletion
                confirmation = st.warning(
                    f"¿Seguro que desea eliminar '{document_name}'?"
                )
                
                

                if st.button("Confirmar"):
                    if delete_doc(document_name):
                        st.success(
                            f"Documento '{document_name}' eliminado exitosamente.",
                            icon="✔",
                        )
                    else:
                        st.error(
                            f"Hubo un error al intentar eliminar '{document_name}'."
                        )
                elif st.button("Cancelar"):
                    st.info(f"Se cancelo la operacion.")
                    st.rerun()

    uploaded_files = st.file_uploader(
        "Sube un nuevo documento", type="txt", accept_multiple_files=True
    )
    if uploaded_files:
        add_files_to_repo(uploaded_files)

        st.write("Archivos añadidos:")
        for uploaded_file in uploaded_files:
            st.write(uploaded_file.name)


if __name__ == "__main__":
    main()
