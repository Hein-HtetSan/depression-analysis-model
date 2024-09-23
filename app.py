
### this file is the main file of the project
import os

import streamlit as st  # import the streamlit lib

from frontend.components.documentation import show_doc
# import the open recorder components and open uploader components
from frontend.components.record import open_recorder
from frontend.components.uploader import open_uploader

# Get the current working directory (i.e., the project folder)
cwd = os.getcwd()

# Construct the path to the README.md file
readme_path = os.path.join(cwd, "README.md")

# Custom CSS for buttons
st.markdown("""
    <style>
    .sidebar-button {
        display: block;
        width: 100%;
        padding: 10px;
        font-size: 16px;
        text-align: center;
        color: white;
        background-color: #007BFF;
        border: none;
        border-radius: 5px;
        margin-bottom: 10px;
        cursor: pointer;
    }
    .sidebar-button:hover {
        background-color: #0056b3;
    }
    .github-button {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 10px;
        font-size: 16px;
        color: white;
        background-color: #333;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
    }
    .github-button:hover {
        background-color: #555;
    }
    .github-icon {
        margin-right: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation
nav_option = st.sidebar.radio("Choose an option:", ["Demo", "Documentation"])

if nav_option == "Documentation":
    st.sidebar.write("### Documentation")
    st.sidebar.write("Here you can find the documentation for the app...")
    # Add your documentation links or content here
    show_doc(readme_path)

elif nav_option == "Demo":
    # Define the app title
    st.title("VoxWisp AI")
    # Add a radio button to choose between file upload or audio recording
    option = st.selectbox(
        "Choose input method",
        ('Upload a music file', 'Record audio')
    )
    # If the user chooses to upload a music file
    if option == 'Upload a music file':
        open_uploader()
    # If the user chooses to record audio via mic
    elif option == 'Record audio':
        open_recorder()
    
    


