#### this is the uploader component of the streamlit user inteface

import os

import pandas as pd
import streamlit as st

from frontend.components.graph import show_graph
# import the prediction function of the model
from frontend.components.predict import prediction


def open_uploader():
    
    # Inject custom CSS to center the file uploader
    st.markdown("""
        <style>
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 50px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Center the file uploader using the CSS class
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    # sub header of this component
    st.subheader("Upload Audio File Here! 🔉")
    # read the upload file from the streamlit
    audio_file = st.file_uploader("Upload a music file", type=["mp3", "wav"])
    # this is just for center the streamlit file uploader
    st.markdown('</div>', unsafe_allow_html=True)

    if audio_file is not None:
        st.audio(audio_file, format='audio/mp3')
        
        # Save the uploaded file to a temporary location
        wav_directory = "temp/"
        os.makedirs(wav_directory, exist_ok=True)
        audio_path = os.path.join(wav_directory, audio_file.name)
        
        # open the audio file
        with open(audio_path, 'wb') as f:
            f.write(audio_file.getbuffer())
        
        # if the button is clicke then start predict
        if st.button("Classify Depression"):
            with st.spinner("Classifying... Please wait."):
                # Call the prediction function
                prediction_result = prediction(audio_path)

            # Display the result after prediction
            st.subheader("Result")
            # Show the graph of audio
            show_graph(audio_path)  # Ensure this function is defined
            st.success(f"Mood: {prediction_result}")
            
            # remove the uploaded files
            # Check if the file exists before attempting to delete it
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"{audio_path} deleted successfully!")
            else:
                print(f"{audio_path} does not exist.")