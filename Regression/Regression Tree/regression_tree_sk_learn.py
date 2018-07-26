import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv('Position_Salaries.csv')
# x = data.iloc[:,1:2].values
# y = data.iloc[:,2:].values

data = pd.read_csv('studentscores.csv')
x = data.iloc[:,0:1].values
y = data.iloc[:,1:].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=0)



from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

 
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# from sklearn.tree import export_graphviz
# import pydotplus
# 
# dot_data = StringIO()
# export_graphviz(regressor, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

print(y)
print(y_pred)
print(regressor.score(x_test,y_test))


plt.scatter(x_test, y_test, c='green', label="regression line")
plt.plot(x_test,y_pred,label="predicted line")
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
