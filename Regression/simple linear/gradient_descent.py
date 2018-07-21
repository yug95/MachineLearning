import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading data from csv file
data = pd.read_csv('studentscores.csv')


#trainning data
x_train =  data.iloc[0:15,0]
y_train = data.iloc[0:15,1]

x_test = data.iloc[15:25,0]
y_test = data.iloc[15:25,1]


def step_gradient(b,m,x_train,y_train,alpha):
    b_gradient = 0
    m_gradient = 0
    N = len(x_train)

    for i in range(0,N):
        b_gradient += -(2/N) * (y_train[i]-((m * x_train[i] + b)))
        m_gradient += -(2/N) * x_train[i] * (y_train[i] -((m * x_train[i] + b)))
    new_b = b - (alpha * b_gradient)
    new_m = m - (alpha * m_gradient)
    return [new_b,new_m]

def gradient_decent():
    learning_rate = 0.0001
    b = 0
    m = 0
    num_iteration =1000

    for i in range(num_iteration):
            b,m = step_gradient(b,m,x_train,y_train,learning_rate)
    return b,m

def R2Test(y_test,Y):
    mean_y_test = np.sum(y_test) / len(y_test)
    r1 = np.sum((y_test - mean_y_test) ** 2)
    r2 = np.sum((Y - mean_y_test) ** 2)
    r_sq = r2 / r1
    return r_sq

[b,m] = gradient_decent()

Y = b + m * x_test
r_sq = R2Test(y_test,Y)
print(r_sq)


plt.scatter(x_test,y_test,c='#ef5423',label="scatter plot")
plt.plot(x_test,Y,label=" gradient regression line")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()
