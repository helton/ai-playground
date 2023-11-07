from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("LangChain Documentation Helper")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


prompt = st.text_input("Prompt", placeholder="Send a message")

if prompt:
    with st.spinner("Generating response..."):
        response = run_llm(prompt, chat_history=st.session_state["chat_history"])
        sources = set([doc.metadata["source"] for doc in response["source_documents"]])
        formatted_response = (
            f"{response['answer']} \n\n {create_sources_string(sources)}"
        )
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(
            (formatted_response, response["answer"])
        )

if st.session_state["chat_answers_history"]:
    for response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(response)
        # for src in sources:
        #     st.markdown(f"Source: {src}")
