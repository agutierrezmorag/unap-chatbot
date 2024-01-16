import json
import time
from datetime import datetime

import pinecone
import pytz
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account
from icecream import ic
from langchain.cache import InMemoryCache
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from st_pages import show_pages_from_config
from streamlit_feedback import streamlit_feedback
from streamlit.testing.v1 import AppTest

from utils import config


def test_chatbot():
    tester = AppTest.from_file("chatbot.py")
    tester.secrets = {
        "OPEN_AI_KEY": config.OPENAI_API_KEY,
        "PINECONE_KEY": config.PINECONE_API_KEY,
        "PINECONE_ENV": config.PINECONE_ENV,
        "PINECONE_INDEX_NAME": config.PINECONE_INDEX_NAME,
        "REPO_OWNER": config.REPO_OWNER,
        "REPO_NAME": config.REPO_NAME,
        "REPO_BRANCH": config.REPO_BRANCH,
        "REPO_URL": config.REPO_URL,
        "REPO_DIRECTORY_PATH": config.REPO_DIRECTORY_PATH,
        "GITHUB_ACCESS_TOKEN": config.GITHUB_ACCESS_TOKEN,
        "FIRESTORE_TEXT_KEY": config.FIRESTORE_TEXT_KEY,
    }

    tester.run()
    assert not tester.exception


if __name__ == "__main__":
    test_chatbot()
