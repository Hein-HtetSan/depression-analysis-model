import os

import joblib

# Define the path to the .pkl file
model_path = os.path.join('backend', 'xgboost_model.pkl')

# Load the model using joblib
@st.cache_resource
def load_model():
    return joblib.load(model_path)

# Load the model only once when the app starts
model = load_model()