import os

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Consultando documentos**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.markdown("### Consulta:")
        self.status.write(query)
        self.status.update(label=f"**Generando respuesta para:** {query}")
        self.status.write("### Fuentes:")

    def on_retriever_end(self, documents, **kwargs):
        for doc in documents:
            source = os.path.splitext(os.path.basename(doc.metadata["source"]))[0]
            self.status.write(f"Extracto de **{source}**")
            self.status.code(doc.page_content)
        self.status.update(label="Respuesta generada", state="complete")
