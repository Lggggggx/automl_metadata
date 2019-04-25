import numpy as np 
import copy 
from sklearn.externals import joblib
from sklearn.linear_model import SGDRegressor 

metadata = np.load('./10_australian_big_metadata.npy')
X = metadata[:, 0:396]
y = metadata[:, 396]

auto_regressor = joblib.load('./automl.joblib')
