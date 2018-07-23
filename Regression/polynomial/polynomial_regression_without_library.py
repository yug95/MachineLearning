import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')

#appending 1 series at start of data for b0 constant
data = np.append(arr=np.ones((10,1)).astype(int),values=data,axis=1)
data = pd.DataFrame(data)


x = data.iloc[:,[0,2]].values
x_plt = data.iloc[:,2].values
y = data.iloc[:,3].values

#square of x feature
x2 = pd.DataFrame(np.square(x[:,1]))

#generating ploynomial matrix of degree =2
poly_x = np.append(arr=x,values=x2,axis=1)


XT = np.transpose(poly_x)
XTX = np.dot(XT,poly_x).astype(float)
IXTX = np.linalg.pinv(XTX)
IXTXXT = np.dot(IXTX,XT)

IXTXXTY = np.dot(IXTXXT,y)

#B = I(x'x) * x' * y
B = IXTXXTY

#y = Bx + Error
pred_y = np.dot(poly_x,B)

print(pred_y)

plt.scatter(x_plt, y, c='green', label="regression line")
plt.plot(x_plt,pred_y,label="predicted line")
plt.xlabel("X parameters")
plt.ylabel("Y parameters")
plt.legend()

plt.show()

