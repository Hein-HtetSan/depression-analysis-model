import os

import librosa
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

# Check the length of the audio
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) < 22050:  # Check if less than 1 second (assuming 22,050 Hz sample rate)
            st.warning("Audio is too short for processing. Please record for longer.")
            return

        # Display the result and show the graph
        st.subheader("Result")
        show_graph(audio_path)

        # Call the prediction function and display the result
        prediction_result = prediction(audio_path)
        st.success(f"Mood: {prediction_result}")
    else:
        st.warning("No audio recorded yet. Please press the button to record.")
