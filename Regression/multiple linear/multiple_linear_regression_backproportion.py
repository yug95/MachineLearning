#Mutliple linear regression using sklearn librery and statsmodel backproportion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,4].values


#encoding of categorical variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()


#avoiding dummy variable trap
x = x[:,1:]


#spliting data into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#fit model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


print(y_pred)
print(np.sum(y_test-y_pred))


#building optimal model with backward algorithm

import statsmodels.formula.api as sm
x= np.append(arr= np.ones((50,1)).astype(int),values=x,axis=1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())
x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())
x_opt = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())

