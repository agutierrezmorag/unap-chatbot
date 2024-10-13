import pandas as pd
import streamlit as st

from document_management.pinecone_management import (
    get_index_data,
    get_or_create_vectorstore,
)

logo_path = "logos/unap_negativo.png"


def get_namespaces():
    """
    Obtiene los nombres de los espacios de nombres del índice Pinecone.

    Returns:
        list: Una lista de los nombres de los espacios de nombres del índice Pinecone.
    """
    index_data = get_index_data()
    namespaces = []
    for namespace in index_data.namespaces:
        namespaces.append(namespace)
    return sorted(namespaces)


@st.cache_data(show_spinner=False)
def do_query(query, namespace, search_type):
    vectorstore = get_or_create_vectorstore(namespace=namespace)
    if search_type == "Similitud":
        fetched_docs = vectorstore.similarity_search(query, k=10, namespace=namespace)
    else:
        fetched_docs = vectorstore.max_marginal_relevance_search(
            query, k=10, namespace=namespace
        )
    data = [doc.__dict__ for doc in fetched_docs]
    df = pd.json_normalize(data)

    if "type" in df.columns:
        df = df.drop(columns=["type"])

    st.dataframe(df, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Consultas de documentos",
        page_icon="🔍",
        layout="wide",
    )
    with st.sidebar:
        st.image(logo_path, use_column_width=True)
    st.header("🔍 Consultas de documentos")
    st.markdown("Esta herramienta permite buscar documentos similares a una consulta.")
    st.markdown("Los resultados se obtienen de la memoria de la IA. ")
    st.info(
        "Los documentos fueron formateados para ser consultados y entendidos por la IA. Puede que no coincidan con el formato original de los documentos o sean incoherentes.",
        icon="📢",
    )

    namespaces = get_namespaces()

    query = st.text_input("Ingrese su consulta aquí", max_chars=100)

    col1, col2 = st.columns(2)
    with col1:
        namespace = st.selectbox("Categoria", namespaces)
    with col2:
        search_type = st.selectbox(
            "Tipo de búsqueda",
            ["Similitud", "MMR"],
            help="**Similitud**: Busca documentos similares a la consulta. \
                **MMR**: Busca documentos similares a la consulta, pero tambien incluye resultados que sean diferentes entre sí.",
        )

    if st.button("Buscar documentos similares", disabled=not query.strip()):
        do_query(query, namespace, search_type)


if __name__ == "__main__":
    main()
