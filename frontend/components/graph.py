import librosa
import matplotlib.pyplot as plt
import streamlit as st


def show_graph(audio_path):
    # Load the audio file using librosa
    y, srr = librosa.load(audio_path, sr=None)
    
    # Plot the waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=srr)
    plt.title("Waveform of Recorded Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Show the waveform plot in Streamlit
    st.pyplot(plt)