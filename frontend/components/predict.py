### this is where model prediction method exist

import pandas as pd
import streamlit as st
import torchaudio
# import model and configured opensimle libs
from frontend.components.config import model, smile

# def the prediction method
def prediction(audio_path):
    
    # Load and process the uploaded audio
    wav, sr = torchaudio.load(audio_path, normalize=True)

    # Extract features using OpenSMILE
    feature = smile.process_signal(wav, sr).values.tolist()[0]
    
    # Convert to DataFrame for prediction
    df = pd.DataFrame([feature], columns=smile.feature_names)# Rename the columns: first column to 'label', the rest to '0', '1', ..., '87'
    # this will be change the pd columns to numeric format
    new_column_names = [str(i) for i in range(len(smile.feature_names))]
    
    # Apply the new column names
    df.columns = new_column_names
    
    # Predict depression states using the trained binary model
    prediction = model.predict(df)
    
    # Update the label mapping for binary classification
    label_mapping = {0: 'No Depression', 1: 'Depression'}
    
    # decode the prediction
    decoded_predictions = [label_mapping[label] for label in prediction]
    
    # return the prediction
    return decoded_predictions[0]