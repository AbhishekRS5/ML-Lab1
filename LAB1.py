#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Abhishek R S-EEE20005
import numpy as np #for working with numerical data in Python
import pandas as pd #for working with data in Python
import matplotlib.pyplot as plt #for creating various types of plots


# In[2]:


def ComputeCost(X,y,theta): #define a function called ComputeCost that takes three arguments: X, y, and theta
    n=len(y)# n is initialized to the length of y
    h=X.dot(theta)
    square_err=(h-y)**2 ## compute the squared error between the predicted values and the actual values
    return 1/(2*m)*np.sum(square_err) #to return the cost of the linear regression
def gradientDescent (X,y,theta,alpha,num_iters):#define a function called gradientDescent
    n=len(y)# n is initialized to the length of y
    J_history=[] # an empty list created
    for i in  range(num_iters):# loop over the specified number of iterations
        h=X.dot(theta)# for predicted values
        error=np.dot(X.transpose(),(h-y))#compute the error between the predicted and actual values
        descent=alpha*(1/m)*error# compute the descent
        theta-=descent#update theta
        J_history.append(ComputeCost(X,y,theta))# adding the current new value to the list
    return theta,J_history# return new parameters


# In[3]:


data=pd.read_csv('ex1data1.txt',header=None)# read the data from the CSV file and store it in a Pandas DataFrame called 'data'
data_n=data.values # convert the Pandas DataFrame to a NumPy array
m=len(data_n[:,0])# set 'm' to the number of rows in the data
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)#adding a column of ones to the first column
y=data_n[:,1].reshape(m,1)#vector 'y' containing the second column of the data
theta=np.zeros((2,1))# initialize the parameter vector 'theta' to zeros
ComputeCost(X,y,theta)# compute the cost
theta,J_history=gradientDescent(X,y,theta,0.01,1500)# optimize the parameter values using gradient descent.
theta[0,0]# print the intercept term
plt.scatter(data[0],data[1])# create a scatter plot of the data
plt.plot(X,theta[1,0]*X+theta[0,0])# create a line plot of the optimized linear regression model
plt.show # show the plot

