#import streamlit as st
from langchain_openai import ChatOpenAI
import tempfile
import os
import httpx
import tiktoken

# Setup token cache dir environment variable
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

client = httpx.Client(verify=False)
#client = httpx.Client(verify=False)

# LLM and Embedding setup
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-gpt-4o",
    api_key="sk-v54iRos9P6bGLnygS7rgoQ",
    http_client=client
)

print(llm.invoke("any story in 5 lines"))
print("hi")
