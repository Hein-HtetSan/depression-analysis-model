
### this file is the main file of the project

import streamlit as st  # import the streamlit lib

# import the open recorder components and open uploader components
from frontend.components.record import open_recorder
from frontend.components.uploader import open_uploader

# Define the app title
st.title("Depression Classifier")

# Add a radio button to choose between file upload or audio recording
option = st.radio(
    "Choose input method",
    ('Upload a music file', 'Record audio')
)

# If the user chooses to upload a music file
if option == 'Upload a music file':
    open_uploader()
# If the user chooses to record audio via mic
elif option == 'Record audio':
    open_recorder()


