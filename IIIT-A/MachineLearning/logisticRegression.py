'''
Building a logistic regression model from scratch.
'''

import numpy as np
import matplotlib.pyplot as plt

'''
y_pred = g(z) #z is the feature*weight matrix ...
z = w1x1 + w2*x2 + w3*x3 + w4*x4 ... so onn

g is the sigmoid function

Then we calculate the loss of the function.

loss function = -1/m np.sum(y*log(y_pred)+(1-y)*log(1-y_pred))

To decrease the loss function we need to take the derivative of the loss function.
finally derivative of the loss function is -> x_i*(y_pred-y)

Gradient Descent
w1 = w1 - alpha*(derivative of loss function)
'''


# So Lets Start 

# weights are first selected as random..

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost_function(X,y,weights):
    Z = np.dot(X,weights)
    return -1/len(X)*(y*np.log(1-sigmoid(Z))+(1-y)*np.log(1- sigmoid(Z)))

def fit(X,y,epochs,lr):
    loss_list = []

    weights = np.random.rand(X.shape[1]) # weights initialized randomly
    N = len(X)

    for _ in range(epochs):
        # loss = -1/N * (np.sum())
        y_pred = sigmoid(np.dot(X,weights))
        weights = weights - lr*(np.dot(X.T,y_pred-y)/N)
        loss_list.append(cost_function(X,y,weights))
    weights = weights
    return weights,loss_list


def predict(X,weights): # these weights are obtained from the above
    z = np.dot(X,weights)
    list = []
    for i in sigmoid(z):
        if i>0.5:
            list.append(1)
        else:
            list.append(0)
    return list


import pandas as pd
data = pd.read_csv("/home/mononoke/ChessDetection/IIIT-A/MachineLearning/hello.csv")
# print(data.head)
# we are given the scores of the 2 examinations and we have to classify wheather he would get the admission or not .

X_train = data.iloc[:20,:-1]
y_train = data.iloc[:20,-1]

X_test = data.iloc[20:,:-1]
y_test = data.iloc[20:,-1]

loss = []
weights,loss = fit(X_train,y_train,20,0.001)
print("This is the thing")
for i in range(10):
    print(predict(X_test,weights))

print(y_test)
print("Accuracy is 50%")

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
print(lr.predict(X_test),end=" ")
print("From sklearn the accuracy is 94%")

# What we used above is the gradient descent and now we would be using the mini batch gradient descent


# Now for adding the regularization term we have to add the beta*np.sum((weights)**2) to the cost function 

# The derivative gets added by the term of (lambda/m)*np.sum(weights) # for batch gradient descent




