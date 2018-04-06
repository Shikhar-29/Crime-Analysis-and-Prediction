# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('43_Arrests_under_crime_against_women.csv')
X = pd.get_dummies(dataset, columns=['Area_Name', 'Group_Name', 'Sub_Group_Name'], drop_first = True)
X = X.drop(['Persons_Trial_Completed'], axis = 1)
Y = dataset.iloc[: , 15].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state = 0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

y_pred = lin_reg.predict(X_test)

plt.scatter(X.Year, Y, color = 'red')
plt.title('year vs trial completed')
plt.show()

C=['2001', '12',	'0',	'32',	'31', 	'2',	'0',	'0',	'59',	'1',	'42',	'73',	'0',	'0',	'0',	'0',	'1',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0', '0',	'1',	'0',	'0',	'0',	'0',	'0', '0',	'0',	'0',	'0',	'0',	'0',	'0',	'0',	'0']
C = np.asarray(C)
C= C.reshape(1, -1)
C = C.astype(float)
V = lin_reg.predict(C)