#%matplotlib inline
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

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
    
def GradientDescent(X,Y,Theta,learning_rate,no_iterations,flag,reg_const):
    # Gradient Descent without Regularization
    if flag==0:
        n = len(X)
        print("_____________________________________________________________________")
        for i in range(no_iterations):
            predict = np.dot(X,Theta)
            loss = predict - Y
            gradient = (np.dot(X.T,loss)*2)/n
            Theta = Theta - (learning_rate*gradient)
            if i % 100 == 0:
                print("After {0} iterations,theta = {1},Loss = {2}\n".format(i,Theta,loss))
    # Gradient Descent with Regularization
    else:

        print("-----------------------------------------")
        print('\033[1;31mRegularization  : \033[1;m',end="")
        print("{0}".format(reg_const))
        print("-----------------------------------------\n")

        n = len(X)
        print("_____________________________________________________________________")
        for i in range(no_iterations):
            predict = np.dot(X,Theta)
            loss = predict - Y
            gradient = (np.dot(X.T,loss)*2)/n + (reg_const/n)*Theta
            Theta = Theta - (learning_rate*gradient)
            if i % 100 == 0:
                print("After {0} iterations,theta = {1},Loss = {2}\n".format(i,Theta,loss))            

    return Theta         

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

def plotter(a,b):

    plt.xlabel('Regularization Constants')
    plt.ylabel('RMSE Values(in x10^5)')
    plt.title('RMSE varying with weights of Regularization')
    #plt.scatter(a,b)
    plt.plot(a,b,'-o')
    #plt.axis(0,0.09)

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
    no_iterations = 1000
    reg_const  = np.arange(0.001,0.01,0.001)
    
    rmse1 = []
    rmse2 = []

    for _reg in reg_const:

       
        Theta_values1 = GradientDescent(X,Y,Theta,learning_rate,no_iterations,0,_reg)
        
        print('\n\033[1;31mTheta : \033[1;m',end="")
        print("{0}".format(Theta_values1))
        predict = np.dot(X_test,Theta_values1)
        print('\033[1;31mRMSE (without Regularization): \033[1;m',end="")
        print("{0}".format(RMSE(Y_test,predict)*(_max - _min)))
        #rmse1.append(RMSE(Y_test,predict)*(_max - _min))

        Theta_values2 = GradientDescent(X,Y,Theta,learning_rate,no_iterations,1,_reg)
        print('\033[1;31mTheta : \033[1;m',end="")
        print("{0}".format(Theta_values2))
        predict = np.dot(X_test,Theta_values2)
        print('\033[1;31mRMSE (with Regularization): \033[1;m',end="")
        print("{0}\n".format(RMSE(Y_test,predict)*(_max - _min)))
        rmse2.append(RMSE(Y_test,predict)*(_max - _min))
        print("____________________________________________________________________")
   

    #print(rmse2)
    plotter(reg_const,rmse2)

if __name__ == "__main__":
    main()