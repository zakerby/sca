import os
import streamlit as st
from streamlit_chat import message

from rag import RAG

st.set_page_config(page_title="Impl RAG", page_icon=":speech_balloon:")

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            try:
                agent_text = st.session_state["robot"].ask(user_text)
            except Exception as e:
                agent_text = f"An error occurred: {e}"

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def process_repository():
    if st.session_state["repo_url"] and len(st.session_state["repo_url"].strip()) > 0:
        repo_url = st.session_state["repo_url"].strip()
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting data"):
            st.session_state["robot"].ingest(repo_url)

        st.session_state["messages"].append((f"Ingested data from {repo_url}", False))


def main():
    if len(st.session_state) == 0:
        st.session_state.messages = []
        st.session_state['robot'] = RAG('phi3')
        
    st.header("RAG Chatbot")
    
    st.session_state["ingestion_spinner"] = st.empty()

    st.text_input("Repository URL", key="repo_url", on_change=process_repository)

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == '__main__':
    main()
