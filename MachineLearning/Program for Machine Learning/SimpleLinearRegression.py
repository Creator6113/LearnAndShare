#Simple Linear Regression
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)
#0.5 = 50% 
# to fit simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set result
#vector of prediction it will contain predicted salary

y_pred = regressor.predict(X_test)
#Visualizing the training set result
# scatter point depicted by this code
plt.scatter(X_train, y_train,color = 'red')
#y_pred is the test salary of x_train
#regression Line
#plotting the regression graph
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(training set)')
#label for x axis
plt.xlabel('Years of Experience')
#label for y axis
plt.ylabel('Salary')
plt.show()

#visualising the test set results
#gaph of test set
plt.scatter(X_test, y_test,color = 'red')
#y_pred is the test salary of x_train
#regression Line
#plotting the regression graph
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(test set)')
#label for x axis
plt.xlabel('Years of Experience')
#label for y axis
plt.ylabel('Salary')
plt.show()
