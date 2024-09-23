import streamlit as st


def show_doc(readme_file):
    with open(readme_file, "r") as f:
        markdown_text = f.read()

    st.markdown(markdown_text)