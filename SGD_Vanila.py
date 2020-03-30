
# Implementation of Vanila SGD 

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Load Boston dataset
def load_data():
    boston = pd.DataFrame(load_boston().data,columns=load_boston().feature_names)
    Y = load_boston().target
    X = load_boston().data
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
    print("\nBoston Dataset:")
    print("X_Train Shape: ",x_train.shape)
    print("X_Test Shape: ",x_test.shape)
    print("Y_Train Shape: ",y_train.shape)
    print("Y_Test Shape: ",y_test.shape)

    # Normalization
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    ## Adding the price Column in the data
    train_data = pd.DataFrame(x_train)
    train_data['y'] = y_train
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train,y_train,x_test,y_test,train_data


# skLearn SGD-based Linear Regression
def sklearn_SGD(x_train, y_train, x_test, y_test, niter):
    model = SGDRegressor(max_iter=niter)
    return model



# SGD for Linear Regression from scratch
def our_SGD(train_data, learning_rate=0.001, niter=1000, batch_size=10):
    
    # Initialize W (vector of coefficients: w1-wd) & w0 (scalar value)
    W = np.zeros(shape=(1,train_data.shape[1]-1))
    w0 = 0
    iteration = 1
    while(iteration <= niter): 
        iteration += 1

        # We will create a small training data set of size K
        temp = train_data.sample(batch_size)
        y = np.array(temp['y'])
        X = np.array(temp.drop('y',axis=1))
        
        # Initialize gradients to 0
        W_gradient = np.zeros(shape=(1,train_data.shape[1]-1))
        w0_gradient = 0
        
        for i in range(batch_size): # Calculating gradients 
            prediction = np.dot(W,X[i]) + w0
            W_gradient = W_gradient + (-2) * X[i] * (y[i]-(prediction))
            w0_gradient = w0_gradient + (-2) * (y[i]-(prediction))
        
        # Updating the weights(W) and Bias(b) with the above calculated Gradients
        W = W - learning_rate * (W_gradient/batch_size)
        w0 = w0 - learning_rate * (w0_gradient/batch_size)
       
    return W,w0 



def predict(X,W,w0):
    y_pred=[]
    for i in range(len(X)):
        y = np.asscalar(np.dot(W,X[i]) + w0)
        y_pred.append(y)
    return np.array(y_pred)



def main():
    x_train,y_train,x_test,y_test,train_data = load_data()
    model = sklearn_SGD(x_train, y_train, x_test, y_test, niter=100)
    model.fit(x_train, y_train)
    y_pred_sksgd = model.predict(x_test)
    print('\nMean Squared Error (SGD of sklearn) :',mean_squared_error(y_test, y_pred_sksgd))


    print('\nGradient Descent from scratch:')
    W,w0 = our_SGD(train_data, learning_rate=0.01, niter=1000, batch_size=10)
    y_pred_our_sgd = predict(x_test, W, w0)
    print('Mean Squared Error (Mini-Batch Gradient Descent) :',mean_squared_error(y_test, y_pred_our_sgd))



if __name__ == "__main__":
    main()
 






















