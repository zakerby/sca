import streamlit as st
from streamlit_chat import message

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



display_messages()
st.text_input("Message", key="user_input", on_change=process_input)