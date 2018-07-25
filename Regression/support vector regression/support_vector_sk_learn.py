#SVR with Sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('Position_Salaries.csv')
x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values
# data = pd.read_csv('studentscores.csv')
# x = data.iloc[:,0:1].values
# y = data.iloc[:,1:].values

#use imputer if any missing value as SVM doesnot support.....
#from sklearn.preprocessing import Imputer
# imputer = preprocessing.Imputer(missing_values='NaN',strategy='mean',axis=0)
# x[:,0:] = imputer.fit_transform(x[:,0:])

#Feature scaling as SVR doesnot apply..

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()

x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)



#fitting svr
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
#regressor = SVR(kernel='linear')
regressor.fit(x,y)
y_pred = regressor.predict(x)

print(y)
print(y_pred)

plt.scatter(x, y, c='green', label="regression line")
plt.plot(x,y_pred,label="predicted line")
plt.xlabel("X parameters")
plt.ylabel("Y parameters")
plt.legend()
plt.show()


