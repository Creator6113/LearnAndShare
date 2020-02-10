#Multiple linear Regrssion
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#LabelEncoder class
labelEncoder_X= LabelEncoder()
# index of our categorical variable is 3 so we encode using 3 ans index
X[: , 3]=labelEncoder_X.fit_transform(X[:, 3])
#Creating Dummy variable
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]
#removed first column of X


# We dont apply feature scaling here as library will take care of it.


#it encode the independent variable so we don nto need it
#labelencoder_y = LabelEncoder()
#y = labelencoder_X.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
#Rgessor is an object of linear regrssion class
regressor = LinearRegression()
#fitting linear regressor object
regressor.fit(X_train,y_train)

#predicting the test result
#with vector y_pred
y_pred= regressor.predict(X_test)


#Backward elemination wegoing to find a team of variables which are having more impact on
#on dependent variable that other
import statsmodels.regression.linear_model as sm
#appending a column of 1 to our data set
# we specified 50 as no of lines and 1 as no of column
 
X = py.append(arr = py.ones((50 ,1 )).astype(int), values = X,axis =1)


#Start Backward Elimination
#making new matrix of feature
# this matrix will contain variable which has high impact on profit

X_opt = X[:,   [0 , 1 , 2 , 3 , 4 , 5]]
#OLS Ordinary least Square
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
















