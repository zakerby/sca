import streamlit as st
from streamlit_chat import message

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_repository():
    if st.session_state["repo_url"] and len(st.session_state["repo_url"].strip()) > 0:
        repo_url = st.session_state["repo_url"].strip()
        repository_name = ''
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting data"):
            st.session_state["robot"].ingest(repo_url)

        st.session_state["messages"].append((f"Ingested data from {repo_url}", False))
        

st.title("Ingest Data")
st.session_state["ingestion_spinner"] = st.empty()
st.text_input("Repository URL", key="repo_url", on_change=process_repository)
