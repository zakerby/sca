import streamlit as st


st.title("Repository analysis")

fig = st.session_state['robot'].plot_embeddings()

st.plotly_chart(fig)