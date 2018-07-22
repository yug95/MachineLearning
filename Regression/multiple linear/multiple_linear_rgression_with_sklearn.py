#Mutliple linear regression using sklearn librery

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


#check with normal way
# x_transpose = np.transpose(x_train)
# XTX = np.dot(x_transpose,x_train)
#
# IXTX = np.linalg.pinv(XTX)
#
# IXTXXT = np.dot(IXTX,x_transpose)
#
# IXTXXTY = np.dot(IXTXXT,y_train)
#
# B = IXTXXTY
#
# pred_y = np.dot(x_test,B)

# print(pred_y)
# print(np.sum(y_test-pred_y))
# # print(y_test)


plt.scatter(x_test[:,2], y_test, c='green', label="predicted line")
plt.scatter(x_test[:,3], y_test, c='blue', label="predicted line")
plt.scatter(x_test[:,4], y_test, c='yellow', label="predicted line")
plt.plot(x_test[:,2],y_pred, 'r', label="regression line")
plt.plot(x_test[:,3], y_pred, 'b', label="regression line")
plt.plot(x_test[:,4],y_pred, 'y', label="regression line")

plt.xlabel("X parameters")
plt.ylabel("Y parameters")
plt.legend()

plt.show()
