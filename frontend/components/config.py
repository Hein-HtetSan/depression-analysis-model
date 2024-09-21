#### this will be use as core method
#### or configuration file there will be config of model, env, database and etc
#### this will be use as a global module file

import opensmile # import the opensmile lib
# import the model
from backend.model import model as ml_model

# Initialize OpenSMILE for feature extraction
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# import the model --> which will be used as a module
model = ml_model