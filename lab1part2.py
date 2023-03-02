#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np #for working with numerical data in Python
import pandas as pd #for working with data in Python
import matplotlib.pyplot as plt #for creating various types of plots
from sklearn.linear_model import LinearRegression # it can be used to create and train a linear regression model on a dataset.


# In[12]:


data=pd.read_csv('ex1data1.txt',header=None)#reads the data from the CSV file into a pandas DataFrame object called data
plt.scatter(data[0],data[1])#creates a scatter plot of the data and plot first column of data in x-axis and second in y axis
data_n=data.values #extract values from the dataframe object and store in numpy array called "data_n"
m=len(data_n[:,0])#m is set to number of rows in data_n
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)#create a numpy array "x" which contain input features 
y=data_n[:,1].reshape(m,1)# y contains the output values, which is the second column of "data_n"

regressor=LinearRegression()#
regressor.fit(X,y)#To create a LinearRegression object called regressor and fit it to the input and output data X and y using the fit()

theta0=regressor.intercept_#
theta1=regressor.coef_#extract the values of the intercept (theta0) and coefficient (theta1) of the linear regression model from the regressor

plt.plot(X,theta1*X+theta0)#plot the linear regression line on the same plot as the scatter plot of the data 
plt.show()#displays the plot on the screen.


# In[ ]:




