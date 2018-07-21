#importing usable library

#numpy for algebra operation
#matplotlib for visualization
#pandas for data analysing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading data from csv file
data = pd.read_csv('studentscores.csv')


#trainning data
x_train =  data.iloc[0:15,0]
y_train = data.iloc[0:15,1]

#test data
x_test =  data.iloc[15:25,0]
y_test = data.iloc[15:25,1]

#mean of x and y training set
mean_x = np.mean(x_train)
mean_y = np.mean(y_train)

def step_gradient(b,m,x_train,y_train,alpha):
    b_gradient = 0
    m_gradient = 0
    N = len(x_train)

    for i in range(0,N):
        b_gradient += -(2/N) * (y_train[i] -((m * x_train[i] + b)))
        m_gradient += -(2/N) * x_train[i]* (y_train[i]-((m * x_train[i] + b)))
    new_b = b - (alpha * b_gradient)
    new_m = m - (alpha * m_gradient)
    return [new_b,new_m]

def gradient_decent():
    learning_rate = 0.001
    b = 0
    m = 0
    num_iteration =1000

    for i in range(num_iteration):
            b,m = step_gradient(b,m,x_train,y_train,learning_rate)
    return [b,m]



#calculate predicted y
def predicted_y(x_train,y_train,mean_x,mean_y):
    # length of training set
    m = len(x_train)

    nume_val = 0;
    deno_val = 0;

    #formula
    """ 
        line = y = b1x + bo
        b1 = slope =((x-x_mean) * (y-y_mean)) / x-x_mean^2
        b0 = intercept = y-b1x
    """


    for i in range(m):
        nume_val += (x_train[i] - mean_x) * (y_train[i] - mean_y)
        deno_val += (x_train[i] - mean_x) ** 2

    b1 = nume_val / deno_val
    b0 = mean_y - (b1 * mean_x)
    print("calculated b={0} and m ={1} without using gradient descent".format(b0, b1))
    Y = b0 + b1 * x_test
    return Y

#calculate R2test for accuracy

def R2Test(y_test,Y):
    mean_y_test = sum(y_test) / len(y_test)
    r1 = sum((y_test - mean_y_test) ** 2)
    r2 = sum((Y - mean_y_test) ** 2)
    r_sq = r2 / r1
    return r_sq


Y = predicted_y(x_train,y_train,mean_x,mean_y)
r_sq = R2Test(y_test,Y)
print("R2 test without gradient descent {r_sq}".format(r_sq=r_sq))
[b,m] = gradient_decent()
print("calculated b={0} and m ={1} using gradient descent".format(b,m))
Y_gradient_pred = b + m * x_test
r_sq_gradient = R2Test(y_test,Y_gradient_pred)
print("R2 test with gradient descent {r_sq_gradient}".format(r_sq_gradient=r_sq_gradient))


plt.scatter(x_test,y_test,c='#ef5423',label="scatter plot")
plt.plot(x_test,Y,label="regression line")
plt.plot(x_test,Y_gradient_pred,label=" gradient regression line")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()

plt.show()


