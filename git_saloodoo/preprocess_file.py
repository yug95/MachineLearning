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


class Indego(BaseEstimator, TransformerMixin):
    
    def fit( self, df, y = None ):
        print("in indego fit")
        return self
    
    def transform(self, df, y = None):
    
        def time_convert(df,time_col):
            df[time_col] = pd.to_datetime(arg=df[time_col], infer_datetime_format=True)
            return df
    
        def drop_column(df, drop_col_name):
            df.drop(columns=[drop_col_name],axis=1,inplace=True)
            return df
        
        def punctuation_cleaning(df,punc_col):
            char_index = df[df[punc_col].str.contains('[A-Za-z]', na=False)].index
            df.drop(char_index,inplace=True)
            return df
        
        def remove_null(df,col_name):
            df = df.loc[~df[col_name].isnull()]
            return df
        
        def convert_data_type(df, col_list,data_type):
            df[col_list] = df[col_list].astype(data_type)
            return df
    
        def station_cleaning(df, station_col_list,final_station_col):
            drop_index = df.loc[(df[station_col_list[0]].isnull()) & (df[station_col_list[1]].isnull())].index
            df.drop(drop_index,inplace=True)
            df[station_col_list] = df[station_col_list].fillna('')
            df[final_station_col] = df[station_col_list[0]].astype(str) +df[station_col_list[1]].astype(str)
            df.drop(columns=station_col_list,inplace=True)
            return df
    
        def lat_lon_cleaning(df,lat_lon_col):
            lat_lon_null = df.loc[(df[lat_lon_col[0]].isnull()) & (df[lat_lon_col[1]].isnull())].index
            df.drop(lat_lon_null,inplace=True)
            df =  remove_null(df,lat_lon_col[1])
            df = remove_null(df,lat_lon_col[0])
            return df
    
        def remove_lat_lon_outlier(df,lat_lon_list):
            df = df.loc[(df[lat_lon_list[0]]!=0) | (df[lat_lon_list[1]]!=0)]
            df = punctuation_cleaning(df,lat_lon_list[0])
            return df
    
        def change_lang_lat_value(df):
            df.loc[df.start_lat <=0,'start_lat'] = abs(df.start_lat)
            df.loc[df.end_lat <=0,'end_lat'] = abs(df.end_lat)
            return df
    
        def degree_to_radion(degree):
            return degree*(np.pi/180)

        def calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):

            from_lat = degree_to_radion(pickup_latitude)
            from_long = degree_to_radion(pickup_longitude)
            to_lat = degree_to_radion(dropoff_latitude)
            to_long = degree_to_radion(dropoff_longitude)

            radius = 6371.01

            lat_diff = to_lat - from_lat
            long_diff = to_long - from_long

            a = np.sin(lat_diff / 2)**2 + np.cos(degree_to_radion(from_lat)) * np.cos(degree_to_radion(to_lat)) * np.sin(long_diff / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            return radius * c
    
        def add_new_date_time_features(dataset):
            dataset['hour'] = dataset.start_time.dt.hour
            dataset['day'] = dataset.start_time.dt.day
            dataset['month'] = dataset.start_time.dt.month
            dataset['year'] = dataset.start_time.dt.year
            dataset['day_of_week'] = dataset.start_time.dt.dayofweek

            return dataset
       
        print("in transformer method")
        #df = time_convert(df,'start_time')
        df.set_index('trip_id',inplace=True)
        df = station_cleaning(df,['start_station','start_station_id'],'start_station_complete')
        df = station_cleaning(df,['end_station','end_station_id'],'end_station_complete')
        df = punctuation_cleaning(df,'bike_id')
        df = remove_null(df,'bike_id')
        df = lat_lon_cleaning(df,['end_lat','start_lat'])
        df = remove_lat_lon_outlier(df,['end_lat','end_lon'])
        df = remove_lat_lon_outlier(df,['start_lat','start_lon'])
        df = convert_data_type(df,['start_lat','start_lon','end_lat','end_lon'],float)
        df = change_lang_lat_value(df)
        df['distance'] = calculate_distance(df.start_lat, df.start_lon, df.end_lat, df.end_lon)
        #df = add_new_date_time_features(df)
        df = convert_data_type(df,['start_station_complete','end_station_complete','bike_id'],float)
        drop_list = ['bike_type','passholder_type','start_time','end_time','end_lat','end_lon','start_lat','start_lon']
        df.drop(columns = drop_list,axis=1,inplace=True)        
        return df
