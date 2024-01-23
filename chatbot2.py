from datetime import datetime
from operator import itemgetter

import pinecone
import streamlit as st
from icecream import ic
from langchain.callbacks.manager import collect_runs
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import format_document
from langchain.schema.runnable import Runnable, RunnableMap
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores import Pinecone
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import config
from utils.callbacks import StreamHandler

pinecone.init(
    api_key=config.PINECONE_API_KEY,
    environment=config.PINECONE_ENV,
)

st.markdown(
    """
    <style>
    #footnotes {
        display: none
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_docs(docs):
    ic(docs)
    return "\n\n".join(doc.page_content for doc in docs)


def _combine_documents(docs, document_separator="\n\n"):
    document_prompt = PromptTemplate.from_template(template="{page_content}")
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def answer(question):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        openai_api_key=config.OPENAI_API_KEY,
        max_tokens=1000,
        streaming=True,
    )
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    vectorstore = Pinecone.from_existing_index(
        index_name=config.PINECONE_INDEX_NAME, embedding=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )
    memory = st.session_state.memory

    template = """Answer the question based only on the following context and the chat's history:
    {context}

    Question: {question}
    
    End every answer with 'YAHOOOO' to end the conversation.
    Answer:
    """
    ingress = RunnableMap(
        {
            "question": RunnablePassthrough(),
            "chat_history": lambda x: memory.load_memory_variables(x)["chat_history"],
            "context": retriever,
        }
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                template,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    chain = ingress | prompt | llm

    return chain


if __name__ == "__main__":
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
            return_messages=True,
        )

    for msg in st.session_state.langchain_messages:
        avatar = "🦜" if msg.type == "ai" else None
        with st.chat_message(msg.type, avatar=avatar):
            st.markdown(msg.content)

    if prompt := st.chat_input(placeholder="Ask me a question!"):
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant", avatar="🦜"):
            chain = answer(prompt)

            message_placeholder = st.empty()
            full_response = ""

            input_dict = {"question": prompt}
            with collect_runs() as cb:
                for chunk in chain.stream(prompt, config={"tags": ["Streamlit Chat"]}):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                st.session_state.memory.save_context(
                    input_dict, {"answer": full_response}
                )
                run_id = cb.traced_runs[0].id
            message_placeholder.markdown(full_response)
