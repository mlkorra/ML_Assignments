
# Part b : Experimenting with Optimization Algorithms
from numpy.linalg import inv
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


# Computing Cost Function
def RMSE(Y,X):
    n = len(X)
    rmse = np.sqrt(sum((Y-X)**2/n))
    return rmse

def Cost(X,Y,Theta):
    
    H = np.dot(X,Theta)
    loss = H-Y
    error = (loss**2)/2*len(Y)
    return error
    
def OptimizationFunction(X,Y,Theta1,Theta2,learning_rate,no_iterations,flag):
    # Gradient Descent

    if flag==0:
        n = len(X)
        Theta = Theta1
        for i in range(no_iterations):
            predict = np.dot(X,Theta)
            #print("X : {0}".format(X.shape))
            #print("Theta : {0}".format(Theta.shape))
            #print("Predict")
            #print(predict.shape)
            #print("Y")
            #print(Y.shape)
            loss = predict - Y
            gradient = (np.dot(X.T,loss)*2)/n
            Theta = Theta - (learning_rate*gradient)
            if i % 100 == 0:
                print("After {0} iterations,theta = {1},Loss = {2}\n".format(i,Theta,loss))

        return Theta        
    # ILRS
    else:
        n = len(X)
        #Theta = Theta2
        #print("Theta 2 :{0} ".format(Theta2.shape))
        for i in range(no_iterations):
            predict1 = np.dot(X,Theta2)
           
            loss1 = predict1 - Y
            gradient = (np.dot(X.T,loss1))
            
            #          ( X.T.dot(W).dot(y) ) )
            Hessian_inverse  = inv(np.dot(X.T,X))
            #print("1 : {0}".format(Theta2.shape))
            #print("her : {0}".format(Hessian_inverse.shape))
            #np.reshape(Hessian_inverse,(5,))
            Theta2 = Theta2 - np.dot(Hessian_inverse,gradient)
            #np.reshape
            #Theta2 = Theta2.reshape(5,)
            #print("2 : {0}".format(Theta2.shape))
            if i % 100 == 0:
                print("After {0} iterations,theta = {1},Loss = {2}\n".format(i,Theta2,loss1))            
    return Theta2       

def FeatureScaler(X):

    n = X.shape[0]
    _min = np.amin(X)
    _max = np.amax(X)
    for i in range(n):
        X[i] = (X[i]-np.amin(X))/(np.amax(X)-np.amin(X))
    #print(X)    
    return X,_min,_max

def FeatureScaler_Test(X,_min,_max):

    n =  X.shape[0]
    for i in range(n):
        X[i] = (X[i]-(_min))/(_max - _min)

    return X

def plotter(a,b,c):

    plt.xlabel('No of iterations')
    plt.ylabel('RMSE Values')
    plt.title('RMSE varying with Num of Iterations')
        
    plt.plot(a,b,label="Gradient Descent",linewidth = 1)
    plt.plot(a,c,label="IRLS",linewidth = 1.5)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

    plt.show()

def main():

    df = pd.read_csv('house_data.csv')
    df.head()
    n = df.shape[0]
    split = int(n*4/5)
    train_set,test_set = df[:split],df[split:]

    # Assigning values from Dataframe
    
    no_sqft1 = train_set['sqft'].values
    no_sqft,_min,_max = FeatureScaler(no_sqft1)
    no_sqft1 = test_set['sqft'].values
    no_sqft_test = FeatureScaler_Test(no_sqft1,_min,_max)
    

    no_floors1 = train_set['floors'].values
    no_floors,_min,_max = FeatureScaler(no_floors1)
    no_floors1 = test_set['floors'].values
    no_floors_test = FeatureScaler_Test(no_floors1,_min,_max)
    

    no_bedrooms1 = train_set['bedrooms'].values
    no_bedrooms,_min,_max = FeatureScaler(no_bedrooms1)
    no_bedrooms1 = test_set['bedrooms'].values
    no_bedrooms_test = FeatureScaler_Test(no_bedrooms1,_min,_max)

 
    no_bathrooms1 = train_set['bathrooms'].values
    no_bathrooms,_min,_max = FeatureScaler(no_bathrooms1)
    no_bathrooms1 = test_set['bathrooms'].values
    no_bathrooms_test = FeatureScaler_Test(no_bathrooms1,_min,_max)

    price1 = train_set['price'].values
    price,_min,_max = FeatureScaler(price1)
    price1 = test_set['price'].values
    price_test = FeatureScaler_Test(price1,_min,_max)

    # Assiging values for Matrices
    n = len(price)
    bias = np.ones(n)
    n = len(price_test)
    bias_test = np.ones(n)

    # Train Values
    X = np.array([bias,no_sqft,no_floors,no_bedrooms,no_bathrooms]).T
    Y = np.array(price)
    Theta = np.random.rand(5)

    # Test Values
    X_test = np.array([bias_test,no_sqft_test,no_floors_test,no_bedrooms_test,no_bathrooms_test]).T
    Y_test = np.array(price_test)

    # Hyper Parameters
    learning_rate = 0.05
    no_iterations = np.arange(0,50,5)

    rmse2 = []
    rmse1 = []

    for _iter in no_iterations:

        print("-----------------------------------------")
        print('\033[1;31mNum of Iterations : \033[1;m',end="")
        print("{0}".format(_iter))
        print("-----------------------------------------\n")

        
        Theta_values1 = OptimizationFunction(X,Y,Theta,Theta,learning_rate,_iter,0)
        print('\033[1;31mTheta : \033[1;m',end="")
        print("{0}".format(Theta_values1))
        predict = np.dot(X_test,Theta_values1)
        print('\033[1;31mRMSE (with Gradient Descent): \033[1;m',end="")
        print("{0}".format(RMSE(Y_test,predict)*(_max - _min)))
        rmse1.append(RMSE(Y_test,predict)*(_max - _min))

        Theta_values2 = OptimizationFunction(X,Y,Theta,Theta,learning_rate,_iter,1)
        print('\033[1;31mTheta : \033[1;m',end="")
        print("{0}".format(Theta_values2))
        predict = np.dot(X_test,Theta_values2)
        print('\033[1;31mRMSE (with IRLS): \033[1;m',end="")
        print("{0}".format(RMSE(Y_test,predict)*(_max - _min)))
        rmse2.append(RMSE(Y_test,predict)*(_max - _min))

    plotter(no_iterations,rmse1,rmse2)

if __name__ == "__main__":
    main()