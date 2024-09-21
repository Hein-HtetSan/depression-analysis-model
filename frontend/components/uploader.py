import os

import librosa
import pandas as pd
import streamlit as st

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
    st.subheader("Upload Audio File Here! 🔉")
    audio_file = st.file_uploader("Upload a music file", type=["mp3", "wav"])
    st.markdown('</div>', unsafe_allow_html=True)

    if audio_file is not None:
        st.audio(audio_file, format='audio/mp3')
        
        # Save the uploaded file to a temporary location
        wav_directory = "temp/"
        os.makedirs(wav_directory, exist_ok=True)
        audio_path = os.path.join(wav_directory, audio_file.name)
        
        with open(audio_path, 'wb') as f:
            f.write(audio_file.getbuffer())
        
        if st.button("Classify"):
            prediction_result = prediction(audio_path)
            st.write(f"Prediction : {prediction_result}")