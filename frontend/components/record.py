import os

import streamlit as st
from audio_recorder_streamlit import audio_recorder

from frontend.components.graph import show_graph
from frontend.components.predict import prediction


# Function to handle audio recording and prediction
def open_recorder():
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

    # Subheader for recording audio section
    st.subheader("Press the button to record audio! 🔉")

    # Use audio-recorder-streamlit to record the audio
    audio_bytes = audio_recorder()

    # If audio is recorded, process it
    if audio_bytes:
        # Play the recorded audio
        st.audio(audio_bytes, format="audio/wav")

        # Save the recorded audio to a temporary file
        wav_directory = "temp/"
        os.makedirs(wav_directory, exist_ok=True)
        audio_path = os.path.join(wav_directory, "recorded_audio.wav")

        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)  # Save audio_bytes to audio_path

        # Display the result and show the graph
        st.subheader("Result")
        show_graph(audio_path)

        # Call the prediction function and display the result
        prediction_result = prediction(audio_path)
        st.success(f"Mood: {prediction_result}")
    else:
        st.warning("No audio recorded yet. Please press the button to record.")
