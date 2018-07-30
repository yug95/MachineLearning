import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('dataset1.csv')
intercept = np.ones((data.shape[0], 1))
data = pd.DataFrame(np.concatenate((intercept, data), axis=1))
X = data.iloc[:,[0,1,2]]
Y = data.iloc[:, -1]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_k = StandardScaler()
sc_y = StandardScaler()
X_train = sc_k.fit_transform(X_train)
X_test = sc_k.fit_transform(X_test)

def Sigmoid(z):
    ex = np.exp(-z)
    G_of_Z = 1.0 / (1.0 + ex)
    return G_of_Z

theta = np.zeros(X_train.shape[1])

alpha = 0.1
iterations = 100

for i in range(100):
    h = np.dot(X_train, theta)
    z = Sigmoid(h)
    gradient = np.dot(X_train.T, (Y_train - h))
    theta = theta - (alpha * gradient)


pred_h = np.dot(X_test, theta)
pred_y = Sigmoid(pred_h)
print(pred_h)

Y_test.reset_index(drop=True,inplace=True)
count = 0
for i in range(len(pred_y)):
    if Y_test[i] == pred_y[i]:
        count = count + 1
print(count)


x_0 = X_train[np.where(Y_train == 0.0)]
x_1 = X_train[np.where(Y_train == 1.0)]
#
#plotting points with diff color for diff label
plt.scatter(x_0[:, 1], x_0[:, 2], c='b', label='y = 0')
plt.scatter(x_1[:, 1], x_1[:, 2], c='r', label='y = 1')
#
#     # plotting decision boundary
# x1 = np.arange(0, 1, 0.1)
# x2 = -(theta[0] + theta[1] * x1 )/theta[2]
# plt.plot(x1, x2, c='k', label='reg line')
# #
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
