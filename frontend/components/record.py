import os

import pandas as pd
import speech_recognition as sr
import streamlit as st

from frontend.components.predict import prediction

# Initialize recognizer class
recognizer = sr.Recognizer()

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
    
    
    st.subheader("Press button to record audio! 🔉")
    if st.button("Start Record"):
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = recognizer.listen(source)
            st.write("Processing...")

            # Try to recognize the speech in the audio
            try:
                # Convert the speech into text (optional, for confirmation)
                text = recognizer.recognize_google(audio)
                st.write("You said:", text)
                
                # Save the audio to a temporary file
                wav_directory = "temp/"
                os.makedirs(wav_directory, exist_ok=True)
                audio_path = os.path.join(wav_directory, "recorded_audio.wav")
                
                with open(audio_path, 'wb') as f:
                    f.write(audio.get_wav_data())

                # Call the predict function and get the prediction result
                prediction_result = prediction(audio_path)
                st.write(f"Prediction: {prediction_result}")

            except sr.UnknownValueError:
                st.write("Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                st.write(f"Could not request results; {e}")
