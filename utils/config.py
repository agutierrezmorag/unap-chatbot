import os

import streamlit as st

# API keys
OPENAI_API_KEY = st.secrets.openai.api_key
PINECONE_API_KEY = st.secrets.pinecone.api_key

# PINECONE data
PINECONE_ENV = st.secrets.pinecone.env
PINECONE_INDEX_NAME = st.secrets.pinecone.index_name

# GitHub repo keys
REPO_OWNER = st.secrets.github.repo_owner
REPO_NAME = st.secrets.github.repo_name
REPO_BRANCH = st.secrets.github.repo_branch
REPO_URL = st.secrets.github.repo_url
REPO_DIRECTORY_PATH = st.secrets.github.directory_path
GITHUB_ACCESS_TOKEN = st.secrets.github.access_token

# Firestore keys
FIRESTORE_TEXT_KEY = st.secrets.firestore.textkey

# LangSmith data
CHAT_ENVIRONMENT = st.secrets.langsmith.chat_environment

os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.langsmith.tracing
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.langsmith.endpoint
os.environ["LANGCHAIN_API_KEY"] = st.secrets.langsmith.api_key
os.environ["LANGCHAIN_PROJECT"] = st.secrets.langsmith.project
