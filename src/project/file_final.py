from joblib import dump
from joblib import load
import pandas as pd
import pickle
from models import Polynomial_regression


with open('file_pickle.pickle','wb') as f:
    pickle.dump(Polynomial_regression.lr_poly,f) # type: ignore

