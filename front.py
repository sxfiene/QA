from pathlib import Path
import streamlit as st
import requests
from io import StringIO

API_URL = "http://localhost:8000"

st.set_page_config(
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

context = ""

genre = st.radio(
    "Choose your model",
    ["BERT", "RoBERTa", "T5"])

uploaded_file = st.file_uploader("Fichier de contexte", type=["txt"])
if uploaded_file is not None:
    context = uploaded_file.read().decode("utf-8")

st.title("Chat Bot w/ FastAPI & Streamlit")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        res = requests.post(f"{API_URL}/question",
                            json={"content": prompt, "context": context, "source": genre.lower()})
        print(res.json())
        st.write(res.json()["content"])
    st.session_state.messages.append(res.json())
