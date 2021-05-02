import json
import pickle
import numpy as np
import pandas as pd
#from sklearn.externals 
import joblib
from azureml.core.model import Model



def init():
    global model
    model_path = Model.get_model_path(model_name = 'frauddetection')
    model = joblib.load(model_path)
    
    
    
def run(data):
    #data = np.array(json.loads(data)['data'])
    result = model.predict()
    
    return result.tolist()
