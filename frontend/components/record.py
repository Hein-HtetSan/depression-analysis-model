import os

import pandas as pd
import speech_recognition as sr
import streamlit as st

from frontend.components.graph import show_graph
# Import the prediction method to predict the input
from frontend.components.predict import prediction

# Initialize recognizer class
recognizer = sr.Recognizer()

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

    # Initialize session state to control "listening" and "processing" phases
    if 'listening' not in st.session_state:
        st.session_state.listening = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Subheader for recording audio section
    st.subheader("Press the button to record audio! 🔉")

    # Button to start recording
    if st.button("Start Record"):
        # Set session state to show "Listening..."
        st.session_state.listening = True
        st.session_state.processing = False
        
        # Update Streamlit UI to show "Listening..." message
        with st.spinner("Listening..."):
            with sr.Microphone() as source:
                print("start listening")
                audio = recognizer.listen(source)
                
                # Set session state for "Processing..."
                st.session_state.listening = False
                st.session_state.processing = True

        # Now show the processing spinner
        with st.spinner("Processing..."):
            print("start processing")
            
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
                    
                st.subheader("Result")
                # show the graph of audio
                show_graph(audio_path)
                
                # Call the prediction function and get the prediction result
                prediction_result = prediction(audio_path)
                st.success(f"Mood: {prediction_result}")
                
                

            except sr.UnknownValueError:
                st.warning("Sorry, I could not understand the audio.")  # Warning for unknown value error
            except sr.RequestError as e:
                st.warning(f"Could not request results; {e}")

        # Reset the processing state after completion
        st.session_state.processing = False

