# Here I am implementing the logistic regression from Scratch with almost no dependencies on any ML libraries to understand the foundational working

import pandas as pd
import numpy as np


def sigmoid(x):
    u = 1 / (1 + np.exp(-x))
    return u
    
class Logistic_Regression():

    def __init__(self,learn_rate,no_iteration):
        self.learn_rate = learn_rate
        self.no_iteration = no_iteration
        self.weight = None
        self.bias = None

    def fit(self,X,y):
        sample, feature = X.shape
        self.weight = np.zeros(feature)
        self.bais = 0

        for _ in range(self.no_iteration):
            linear_func = np.dot(X,self.weight) + self.bais
            predict = sigmoid(linear_func)

            dw = (1/sample) * np.dot(X.T,predict - y)
            db = (1/sample) * np.sum(predict - y)
     

            self.weight = self.weight - self.learn_rate*dw
            self.bias = self.bais - self.learn_rate*db

    def pred(self,X):
        predict = np.dot(X,self.weight) + self.bais
        y_predict = sigmoid(predict)
        class_pred = [0 if y < 0.5 else 1 for y in y_predict]
        return class_pred

