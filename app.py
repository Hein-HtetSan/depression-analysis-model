
### this file is the main file of the project
import streamlit as st  # import the streamlit lib

# import the open recorder components and open uploader components
from frontend.components.record import open_recorder
from frontend.components.uploader import open_uploader

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

# Define the app title
st.title("Depression Classifier")


# Create a button for GitHub
github_url = "https://github.com/Hein-HtetSan/depression-analysis-model.git"  # Replace with your GitHub link
if st.button("🍵 Source Code"): # Redirect to the GitHub link
    st.write(f"[Click here to view the Source Code]({github_url})")

# Sidebar for navigation
nav_option = st.sidebar.radio("Choose an option:", ["Demo", "Documentation"])

if nav_option == "Documentation":
    st.sidebar.write("### Documentation")
    st.sidebar.write("Here you can find the documentation for the app...")
    # Add your documentation links or content here

elif nav_option == "Demo":
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
    
    


