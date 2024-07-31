import os
import streamlit as st

from rag import RAG

 
def main():
    st.set_page_config(page_title="Impl RAG", page_icon=":speech_balloon:")
    st.title("SCA: Streamlit Code Analysis")
        
    st.sidebar.success("Select a page above.")
    
    # Init state of the application
    if len(st.session_state) == 0:
        st.session_state.messages = []
        st.session_state['robot'] = RAG('phi3')
        st.session_state["ingestion_spinner"] = st.empty()


    
    st.text("This basic app, aims to show how to use the RAG model in a Streamlit app.")
    st.text("First, you have to ingest a repository on the ingest page, then you can ask questions about the code in the chat page.")

if __name__ == '__main__':
    main()
