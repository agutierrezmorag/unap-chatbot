import os
from typing import Any, Dict, List

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from langchain_community.callbacks.openai_info import get_openai_token_cost_for_model
from langchain.schema.messages import BaseMessage
from tiktoken import encoding_for_model


class TokenUsageTrackingCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tiktoken = encoding_for_model(model_name)
        self.tokens_sent = 0

    def always_verbose(self) -> bool:
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *args, **kwargs
    ) -> Any:
        self.tokens_sent += sum(
            [len(self.tiktoken.encode(prompt)) for prompt in prompts]
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        self.tokens_sent += sum(
            [len(self.tiktoken.encode(prompt[0].content)) for prompt in messages]
        )

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        tokens_received = sum(
            [
                len(self.tiktoken.encode(g.text))
                for generations in response.generations
                for g in generations
            ]
        )
        input_token_cost = get_openai_token_cost_for_model(
            model_name=self.model_name, num_tokens=self.tokens_sent
        )
        output_token_cost = get_openai_token_cost_for_model(
            model_name=self.model_name, num_tokens=tokens_received, is_completion=True
        )
        total_cost = input_token_cost + output_token_cost
        print(
            f"\n[{self.model_name}]\n\tTokens sent: {self.tokens_sent}\n\tTokens received: {tokens_received}\nTotal Cost (USD): ${total_cost}"
        )


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.tokens = 0
        self.question = 0

    #Solucion a lo bruto a modo de parche, se seguira buscando una solucion mas optima.
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        
        if(token =="¿" or token =="?" and self.question != 2):
            self.question +=1
        elif(self.question != 2 ):
            pass
        else:
            self.text += token
            self.container.markdown(self.text)
            self.tokens += 1


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
