import os

import streamlit as st

# OPENAI data
OPENAI_CHAT_MODEL = st.secrets.openai.chat_model
OPENAI_EMBEDDING_MODEL = st.secrets.openai.embedding_model
OPENAI_API_KEY = st.secrets.openai.api_key

# PINECONE data
PINECONE_API_KEY = st.secrets.pinecone.api_key
PINECONE_INDEX_NAME = st.secrets.pinecone.index_name
PINECONE_INDEX_DIMENSIONS = st.secrets.pinecone.index_dimensions
PINECONE_INDEX_HOST = st.secrets.pinecone.index_host

# GitHub repo keys
REPO_OWNER = st.secrets.github.repo_owner
REPO_NAME = st.secrets.github.repo_name
REPO_BRANCH = st.secrets.github.repo_branch
REPO_URL = st.secrets.github.repo_url
REPO_DIRECTORY_PATH = st.secrets.github.directory_path
GITHUB_ACCESS_TOKEN = st.secrets.github.access_token

# Google Cloud keys
AI_STUDIO_API_KEY = st.secrets.google.ai_studio_api_key

# Firestore keys
FIRESTORE_TEXT_KEY = st.secrets.firestore.textkey

# LangSmith data
LANGCHAIN_TRACING_V2 = st.secrets.langsmith.tracing
LANGCHAIN_ENDPOINT = st.secrets.langsmith.endpoint
LANGCHAIN_API_KEY = st.secrets.langsmith.api_key
LANGCHAIN_PROJECT = st.secrets.langsmith.project
CHAT_ENVIRONMENT = st.secrets.langsmith.chat_environment

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
