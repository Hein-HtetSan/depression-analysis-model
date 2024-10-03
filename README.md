# VoxWisp AI: AI-Powered Depression Detection

## Introduction


VoxWispAI is a cutting-edge platform that detects depression through voice analysis, using advanced XGBoost machine learning techniques. Developed by Simbolo students, it aims to bridge the gap between human emotions and AI, addressing emotional and social challenges in today's society.



## Contributors 

- Pyi Bhone Kyaw
- Hein Htet San
- Hein Htet Aung (David Chang)

## Supervisor

- Tr. Htet Htet Mon

## Institution

- Simbolo

## Project Overview


Depression Detection is a speech-based classifier that analyzes emotional and acoustic features to detect depression. We use OpenSMILE to extract eGeMAPS features, which are then classified by the XGBoost algorithm to predict depression.


 
Our project, VoxWispAI, uses the RAVDNESS dataset to extract 88 audio features using OpenSMILE's eGeMAPSV02 configuration. These features are then mapped to binary labels and fed into the XGBoost model to predict depression outcomes, enabling VoxWispAI to identify emotional states from audio characteristics.


## Dataset Preparation


We used the RAVDESS dataset, which contains 1440 WAV files of emotional speech and song by 24 actors. The dataset features eight emotions, including neutral, happy, sad, and angry, expressed through both speech and song with varying intensities.


### Dataset Details

- Modality: full-AV, video-only, audio-only
- Vocal channel: speech, song
- Emotion: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Emotional intensity: normal, strong (except for neutral)
- Statement: "Kids are talking by the door", "Dogs are sitting by the door"
- Repetition: 1st repetition, 2nd repetition
- Actor: 24 actors (odd numbers: male, even numbers: female)

## Feature Extraction


Effective feature extraction is crucial for accurately identifying depression through Speech Emotion Recognition (SER). This process converts raw audio (.wav) files into numerical values that capture various emotional and acoustic characteristics. 


### OpenSMILE and eGeMAPS

| Tool | Description |
|---|---|
| **OpenSMILE** | openSMILE (open-source Speech and Music Interpretation by Large-space Extraction) is a complete and widely-used open-source software for feature extraction from speech and music signals. |
| **eGeMAPS** | eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set) is an open-source acoustic feature set that extracts 88 acoustic parameters from speech signals, including spectral, cepstral, and prosodic features. |

### Code

```python
# Load the necessary Libs
import opensmile
import os
import pandas as pd

# init the openSMILE feature extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# Load the audio files
# Path to the subdirectory where audio files are located
sub_dir = 'backend/dataset/Revdess-dataset/Revdess-dataset'
files = os.listdir(sub_dir)
print("Files in subdirectory:", files)
    
audio_files = [os.path.join(sub_dir, file) for file in files if file.endswith('.wav')]
print("Audio files:", audio_files)


# feature extraction start
features = []

for file in audio_files:
    feature_vector = smile.process_file(file)
    feature_vector['filename'] = os.path.basename(file)  # Add the filename to the feature vector
    features.append(feature_vector)

# Convert the list of DataFrames to a single DataFrame
features_df = pd.concat(features, ignore_index=True)

# save the extracted features
features_df.to_csv('ravdess_features.csv', index=False)
```

### Merged the feature and label


We used OpenSMILE to extract 88 audio features using the eGeMAPSV02 configuration. These features are then mapped to binary labels indicating the presence or absence of depression. We merged the lable and feature by traditional way.



### Split the dataset

We employed the Ravdess dataset, intentionally divided into
- **60% training subset** (600 samples)
- **20% validation subset** (200 samples)
- **20% testing subset** (200 samples)

with a **random state of 42**

### Code

```python
# load the sklearn
from sklearn.model_selection import train_test_split
# split the dataset
X_train, X_valid, y_train, y_valid = train_test_split(data1.loc[:, data1.columns != 'label'], data1[['label']], test_size=0.2, random_state=42)

```

## Model Architecutre

![Model Architecutre image](https://github.com/Hein-HtetSan/depression-analysis-model/blob/main/image-3.png)


Our project, VoxWispAI, utilizes the RAVDESS dataset to analyze audio features for depression detection. The model architecture is as follows:


- **Feature Extraction**: OpenSMILE extracts 88 audio features using the eGeMAPSV02 configuration.
- **Feature Mapping**: Features are mapped to binary labels indicating the presence or absence of depression.
- **Model**: XGBoost algorithm is used for binary classification to predict depression outcomes.


This systematic approach allows VoxWispAI to effectively identify and classify emotional states based on audio characteristics.

## Training Model


We utilized binary classification with the XGBoost model to distinguish between depressed and non-depressed states based on the extracted features from the RAVDESS dataset. The XGBoost model is effective for depression detection and contributes to the advancement of speech emotion recognition systems.


### Code

```python
# load the necesarry libs
from xgboost import XGBClassifier
import xgboost as xgb

# define param 
params = {
    'objective': 'binary:logistic',  # Binary classification objective
    'booster': 'gbtree',
    'eta': 0.1,                     # Learning rate
    'max_depth': 5,                  # Maximum depth of a tree
    'subsample': 0.7,                # Subsample ratio of the training instances
    'colsample_bytree': 0.8,         # Subsample ratio of columns when constructing each tree
    'seed': 42,                      # Random seed for reproducibility
    'n_estimators': 600,             # Number of boosting rounds (trees)
    'gamma': 0,
    'min_child_weight': 2,
    'colsample_bylevel': 0,
    'colsample_bynode': 1,
    'alpha': 2,                      # L1 regularization term
    'lambda': 2,                     # L2 regularization term

}

# init the model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_valid, label=y_valid)
model = xgb.XGBClassifier(**params)

# train the model
model.fit(X_train, y_train,verbose = True)
```

### XGBoost

![xgboost diagram](https://github.com/Hein-HtetSan/depression-analysis-model/blob/main/image.png)


XGBoost is a highly efficient and scalable gradient boosting algorithm known for its speed and performance in machine learning tasks. It employs a parallelized tree-boosting approach, adding decision trees that predict the residuals of previous predictions, which enhances predictive accuracy.


### Key Features

* **Supports both regression and classification**
* **Includes hyperparameters for model tuning, L1 and L2 regularization to combat overfitting, and robust handling of missing data**
* **Built-in cross-validation and efficient tree-pruning methods contribute to high accuracy and improved convergence**

### Parameter Tuning

|     | Parameter         | Description                                                                | Value           |
| --- | ----------------- | -------------------------------------------------------------------------- | --------------- |
| 0   | objective         | Binary classification objective                                            | binary:logistic |
| 1   | booster           | Type of boosting model                                                     | gbtree          |
| 2   | eta               | Learning rate                                                              | 0.1             |
| 3   | max_depth         | Maximum depth of a tree                                                    | 5               |
| 4   | subsample         | Subsample ratio of the training instances                                  | 0.7             |
| 5   | colsample_bytree  | Subsample ratio of columns when constructing each tree                     | 0.8             |
| 6   | seed              | Random seed for reproducibility                                            | 42              |
| 7   | n_estimators      | Number of boosting rounds (trees)                                          | 600             |
| 8   | gamma             | Minimum loss reduction required to make a further partition on a leaf node | 0               |
| 9   | min_child_weight  | Minimum sum of instance weight (hessian) needed in a child node            | 2               |
| 10  | colsample_bylevel | Subsample ratio of columns for each level                                  | 0               |
| 11  | colsample_bynode  | Subsample ratio of columns for each node                                   | 1               |
| 12  | alpha             | L1 regularization term                                                     | 2               |
| 13  | lambda            | L2 regularization term                                                     | 2               |


The XGBoost model parameters are tailored for effective binary classification. We set the objective to binary:logistic, using decision trees as the booster, with the following settings:


- Learning rate: 0.1
- Max tree depth: 5
- Subsample ratio: 0.7
- Colsample_bytree: 0.8
- Boosting rounds: 600
- Seed: 42 for reproducibility
- Regularization: gamma, min_child_weight, alpha, and lambda to prevent overfitting

## Testing the Model

### Import the model 

```python
import os
import joblib

# Define the path to the .pkl file
model_path = os.path.join('backend', 'xgboost_model.pkl')
# Load the model using joblib
def load_model():
    return joblib.load(model_path)
# Load the model only once when the app starts
model = load_model()
```

### Load the audio

```python
# laod the audio file or
wav_directory = "temp/"
audio_path = os.path.join(wav_directory, audio_file.name)


# record the audio file
import speech_recognition as sr
# Initialize recognizer
recognizer = sr.Recognizer()
# Use the microphone to record
with sr.Microphone() as source:
    print("Please speak:")
    # Listen until speech is complete
    audio = recognizer.listen(source)
# Save the audio to a file (optional)
with open("recorded_audio.wav", "wb") as file:
    file.write(audio.get_wav_data())
```

### Pass to the model

```python
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
```

## Evaluation


We fine-tuned the model to improve accuracy and avoid overfitting. This led to significant improvements in metrics, making our XGBoost model reliable for detecting depressive states in speech.


| Confusion Matrix | ROC Curve |
|----|---|
|  ![confusion matrix diagram](https://github.com/Hein-HtetSan/depression-analysis-model/blob/main/image-1.png) | ![Roc curve diagram](https://github.com/Hein-HtetSan/depression-analysis-model/blob/main/image-2.png)|


Our ROC curve shows strong predictive performance with an AUC of 0.90. This indicates we effectively distinguish between positive and negative classes with minimal trade-offs between true positives and false positives.


## Classification Report and Test Accuracy


The classification report for VoxWispAI, powered by XGBoost, shows strong performance.


- Class 0:
    - Precision: 0.84
    - Recall: 0.83
    - F1-Score: 0.83
- Class 1:
    - Precision: 0.86
    - Recall: 0.87
    - F1-Score: 0.87
- Overall:
    - Accuracy: 85%
    - Macro Average:
      - Precision: 0.85
      - Recall: 0.85
      - F1-score: 0.85
    - Weighted Average:
      - Precision: 0.85
      - Recall: 0.85
      - F1-score: 0.85

## Conclusion:

VoxwispAI, developed by Simbolo students, bridges the gap between artificial intelligence and emotional wellbeing through AI-powered voice depression detection. Guided by ethical principles of **privacy**, **fairness**, and **transparency**, VoxwispAI addresses emotional and social challenges in today's society.

Looking ahead, VoxwispAI aims to expand its capabilities by:
* Incorporating **multimodal inputs** like facial expressions
* Enhancing the detection of various depression states

The project has achieved **86.5% accuracy** in classifying emotional states from 1,440 WAV files using the XGBoost model on the RAVDESS dataset.

Future improvements will focus on:
* Refining **feature extraction**
* Increasing the **dataset size** to boost accuracy and reduce classification errors

While maintaining a commitment to **user trust** and **emotional well-being**, VoxwispAI is poised to make a meaningful impact in the field of AI-powered depression detection.

## References

This project is referenced by the paper - [IEEE ICCE-TW 2024](https://www.icce-tw.org/) **(Proceeding)**

You can try with the model **[Demo](https://depression-analysis-model-lzxgrcdxudwjarhssproas.streamlit.app/)**.

-----

# Try this Model
You can try this model by running the following code in a Python environment with the necessary libraries.

### Project Structure

If you want to check the notebook, you can find here `backend/pre-model.ipynb`

```
- asset/
  - image/
    - __init__.py
- backend/
  - __pycache__/
  - dataset/
    - __init__.py
    - model.py
    - pre-model.ipynb
    - ravdess_features.csv
    - xgboost_model.pkl
- frontend/
  - __pycache__/
  - components/
    - __init__.py
- temp/
  - recorded_audio.wav
- venv/
- .gitignore
- app.py
- image.png
- packages.txt
- README.md
- requirements.txt

```

### Clone the project
```shell
git clone https://github.com/Hein-HtetSan/depression-analysis-model.git

cd depression-analysis-model
```

### Create Virtual Environment
```shell
python -m venv venv
```

**Don't forget to select your kernel environment to run python**

### Activate environment
```shell
# window
./venv/Scripts/activate

# mac
source venv/bin/activate
```

### Install the required libraries
```shell
pip install -r requirements.txt
```

### Run the application
```shell
streamlit run app.py
```
