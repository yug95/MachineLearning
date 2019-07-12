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
import pickle

import warnings
warnings.filterwarnings("ignore")

import preprocess_file
import label_encoder



def data_load(path):
    
    files = glob.glob(r"%s\*.csv"%(path))
    
    for i,file in enumerate(files):
        if i == 0 :
            df = pd.read_csv(file)
        else:
            df = pd.concat([df,pd.read_csv(file)],axis=0)
    
    return df


def model_build():
    df = data_load('data')
    preprocess_pipeline = Pipeline(steps=[('preprocess',preprocess_file.Indego()),('label',label_encoder.custom_label_encoder())])
    dataset = preprocess_pipeline.fit_transform(df) 
                                   
    X = dataset.drop(columns=['duration'])
    y = dataset['duration']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
    
    print(X_test)
    y_train = y_train /60
    y_test =  y_test / 60
    
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)
	
    model = Pipeline(steps=[('Normalizer',Normalizer()),('decision tree', DecisionTreeRegressor())])
    model.fit(X_train,y_train)
    filename = 'finalized_model.pkl'
    pickle.dump(model,open(filename,"wb"))