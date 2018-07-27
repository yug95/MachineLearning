#Random forest implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataset = pd.read_csv('temps.csv')
# x = dataset.iloc[:,[0,1,2,4,5,6,7,8,9,10]]
# y = dataset.iloc[:,-1:]


data = pd.read_csv('studentscores.csv')
x = data.iloc[:,0:1].values
y = data.iloc[:,1:].values

#splitting data into training and test data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=0)



from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x_test,y_test)

pred_y = regressor.predict(x_test)

plt.scatter(x_test, y_test, c='green', label="regression line")
plt.plot(x_test,pred_y,label="predicted line")
plt.xlabel("X parameters")
plt.ylabel("Y parameters")
plt.legend()
plt.show()


#visualizing in larger dimension....

X_grid = np.arange(min(x_test),max(x_test),0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(x_test, y_test, c='green', label="regression line")
plt.plot(X_grid,regressor.predict(X_grid),label="predicted line")
plt.xlabel("X parameters")
plt.ylabel("Y parameters")
plt.legend()
plt.show()
