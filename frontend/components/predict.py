import librosa
import pandas as pd
import streamlit as st

# import model and configured opensimle libs
from frontend.components.config import model, smile


# def the prediction method
def prediction(audio_path):
    try:
        # Load and process the uploaded audio using librosa
        wav, sr = librosa.load(audio_path, sr=None)

        # Extract features using OpenSMILE
        feature = smile.process_signal(wav, sr).values.tolist()[0]

        # Convert to DataFrame for prediction
        df = pd.DataFrame([feature], columns=smile.feature_names)

        # Rename columns for the DataFrame
        new_column_names = [str(i) for i in range(len(smile.feature_names))]
        df.columns = new_column_names

        # Predict depression states using the trained binary model
        prediction = model.predict(df)

        # Map the prediction to human-readable labels
        label_mapping = {0: 'No Depression', 1: 'Depression'}
        decoded_predictions = [label_mapping[label] for label in prediction]

        # Return the prediction
        return decoded_predictions[0]

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None
