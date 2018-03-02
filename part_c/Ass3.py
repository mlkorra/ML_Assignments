import numpy as np
import pandas as pd

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
    
def GradientDescent(X,Y,Theta,learning_rate,no_iterations,flag):

    if flag==1:
        n = len(X)
        for i in range(no_iterations):
            predict = np.dot(X,Theta)
            loss = predict - Y
            gradient = (np.dot(X.T,loss)*2)/n
            Theta = Theta - (learning_rate*gradient)
            if i % 100 == 0:
                print("After {0} iterations,theta = {1},Loss = {2}\n".format(i,Theta,loss))
    
    # Gradient Descent with Regularization

    if flag==2:
        X_square = X**flag
        Y_square = Y**flag
        n = len(X_square)
        for i in range(no_iterations):
            predict = np.dot(X_square,Theta)
            loss = predict - Y_square
            gradient = (np.dot(X_square.T,loss)*2)/n
            Theta = Theta - (learning_rate*gradient)
            if i % 100 == 0:
                print("After {0} iterations,theta = {1},Loss = {2}\n".format(i,Theta,loss))           
    
    if flag==3:
        X_cube = X**flag
        Y_cube = Y**flag
        n = len(X_cube)
        for i in range(no_iterations):
            predict = np.dot(X_cube,Theta)
            loss = predict - Y_cube
            gradient = (np.dot(X_cube.T,loss)*2)/n
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

def plotter(a,b,c,d):

    plt.xlabel('Learning Rate')
    plt.ylabel('RMSE Values')
    #plt.title('RMSE varying with Learning Rates for different Combination of Features\n')
    #linear_comb, = plt.plot(a,b,label="Linear_Comb",linewidth = 1)
    #quad_comb, =   plt.plot(a,c,label="Quadratic_Comb",linewidth = 1.5)
    #cubic_comb, =  plt.plot(a,d,label="Cubic_Comb",linewidth = 2)
    plt.plot(a,b,label="Linear_Comb",linewidth = 1)
    plt.plot(a,c,label="Quadratic_Comb",linewidth = 1.5)
    plt.plot(a,d,label="Cubic_Comb",linewidth = 2)
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    #plt.legend(handles=[linear_comb], loc=0)
    #plt.legend(handles=[quad_comb],loc=1)
    #plt.legend(handles=[cubic_comb],loc=2)

    plt.show()

def main():

    df = pd.read_csv('house_data.csv')
    df.head()
    n = df.shape[0]
    split = int(n*4/5)
    train_set,test_set = df[:split],df[split:]

    # Assigning values from Dataframe
    no_sqft1 = train_set['sqft'].values
    #print(no_sqft1.shape)
    no_sqft,_min,_max = FeatureScaler(no_sqft1)
    #print(no_sqft.shape)
    no_sqft1 = test_set['sqft'].values
    #print(no_sqft1.shape)
    no_sqft_test = FeatureScaler_Test(no_sqft1,_min,_max)
    #print(no_sqft_test.shape)

    #,_min,_max
    #print(no_sqft.shape)
    #no_sqft_test = FeatureScaler_Test(no_sqft,_min,_max)
    #print(no_sqft_test.shape)
    no_floors1 = train_set['floors'].values
    no_floors,_min,_max = FeatureScaler(no_floors1)
    no_floors1 = test_set['floors'].values
    no_floors_test = FeatureScaler_Test(no_floors1,_min,_max)
    #,_min,_max
    #print(no_floors.shape)

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

    # Hyper Parameters
    learning_rate = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
    no_iterations = 1000

    # Altering learning_rates
    rmse1 = []
    rmse2 = []
    rmse3 = []

    for lr in learning_rate:

        print("-----------------------------------------")
        print('\033[1;31mLearning Rate : \033[1;m',end="")
        print("{0}".format(lr))
        print("-----------------------------------------\n")

        Theta_values1 = GradientDescent(X,Y,Theta,lr,no_iterations,1)
        print('\033[1;31mTheta : \033[1;m',end="")
        print("{0}".format(Theta_values1))
        predict = np.dot(X_test,Theta_values1)
        print('\033[1;31mRMSE (with Linear Combinations): \033[1;m',end="")
        print("{0}".format(RMSE(Y_test,predict)*(_max - _min)))
        rmse1.append(RMSE(Y_test,predict)*(_max - _min))

        Theta_values2 = GradientDescent(X,Y,Theta,lr,no_iterations,2)
        print('\033[1;31mTheta : \033[1;m',end="")
        print("{0}".format(Theta_values2))
        predict = np.dot(X_test,Theta_values2)
        print('\033[1;31mRMSE (with Quadractic Combinations): \033[1;m',end="")
        print("{0}\n".format(RMSE(Y_test,predict)*(_max - _min)))
        rmse2.append(RMSE(Y_test,predict)*(_max - _min))
        
        no_iterations = 51
        Theta_values3 = GradientDescent(X,Y,Theta,lr,no_iterations,3)
        print('\033[1;31mTheta : \033[1;m',end="")
        print("{0}".format(Theta_values3))
        predict = np.dot(X_test,Theta_values3)
        print('\033[1;31mRMSE (with Cubic Combinations): \033[1;m',end="")
        print("{0}\n".format(RMSE(Y_test,predict)*(_max - _min)))
        rmse3.append(RMSE(Y_test,predict)*(_max - _min))

    plotter(learning_rate,rmse1,rmse2,rmse3)
        
if __name__ == "__main__":
    main()