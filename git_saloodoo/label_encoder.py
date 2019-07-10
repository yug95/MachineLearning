'''
Importing important packages required for operation..
'''
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,Normalizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

class custom_label_encoder(BaseEstimator,TransformerMixin):
    def fit( self, df, y = None ):
        print("in label encoder fit.....")
        return self
    
    def transform(self, df, y = None):
        print("in label encoder transformer")
        label = LabelEncoder()
        df['trip_route'] = label.fit_transform(df['trip_route_category'])
        df.drop(columns='trip_route_category',inplace=True)
        return df