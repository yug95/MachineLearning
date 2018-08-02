#logistic classifier with multi class in skelarn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('glass.csv')
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

#splitting dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=0)

# #feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

#
# #import logistic library
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class='multinomial',solver='newton-cg',random_state=0)
classifier.fit(X_train,Y_train)

pred_y = classifier.predict(X_test)
# print(classifier.score(X_test,Y_test))
# print(classifier.predict_log_proba(X_test))
# print(pred_y)
# print(Y_test)

Y_test.reset_index(drop=True,inplace=True)
# print(len(Y_test))
count = 0
for i in range(len(Y_test)):
    if pred_y[i] == Y_test[i]:
        count = count +1

# print(count)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,pred_y)
