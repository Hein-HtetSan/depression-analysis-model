import opensmile

from backend.model import model as ml_model

# Initialize OpenSMILE for feature extraction
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

model = ml_model