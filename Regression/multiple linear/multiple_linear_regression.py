import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')

ones = pd.DataFrame(np.ones((40,1)))
test_ones = pd.DataFrame(np.ones((12,1)))
y_train = data.iloc[0:40,1]
x_train = data.iloc[0:40,0:3]
y_test = data.iloc[35:47,1]
x_test = data.iloc[35:47,0:3]

x_test = x_test.reset_index(drop=True)
x_train = ones.join(x_train)
x_test = test_ones.join(x_test)


x_test.columns = range(x_test.shape[1])

x_transpose = np.transpose(x_train)
XTX = np.dot(x_transpose,x_train)

IXTX = np.linalg.pinv(XTX)

IXTXXT = np.dot(IXTX,x_transpose)

IXTXXTY = np.dot(IXTXXT,y_train)

B = IXTXXTY

pred_y = np.dot(x_test,B)
print(pred_y)
print(y_test)

# # Fitting Simple Linear Regression to the Training set
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(x_train, y_train)
# 
# # Predicting the Test set results
# y_pred = regressor.predict(x_test)
# print(y_pred)

#
#
#diff = np.sum(y_test-pred_y)
# diff_sqr = diff * diff
# sum_diff_sqr = sum(diff_sqr)
# cost = sum_diff_sqr * 0.5 * len(Y)
#
# print(cost)
#
# mean_y_test = sum(Y)/len(Y)
#
# r1 = sum((Y - mean_y_test)**2)
# r2 = sum((pred_y - mean_y_test)**2)
#
# r_sq = r2/r1
# print(r_sq)
#
#
#
# print(x_test['GarageArea'].index.get_value(1300))

plt.scatter(x_test[1], y_test, c='green', label="predicted line")
plt.scatter(x_test[2], y_test, c='blue', label="predicted line")
plt.scatter(x_test[3], y_test, c='yellow', label="predicted line")
plt.plot(x_test[1],pred_y, 'r', label="regression line")
plt.plot(x_test[2], pred_y, 'b', label="regression line")
plt.plot(x_test[3],pred_y, 'y', label="regression line")

plt.xlabel("X parameters")
plt.ylabel("Y parameters")
plt.legend()

plt.show()
