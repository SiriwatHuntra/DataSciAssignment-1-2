import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


#Import data
def read_data_from(data_link):
    data =pd.read_csv(data_link)
    #print(data.columns) 
    X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
    Y = data['class']

def test_model():
    r_loop = 30
    seed = 1
    for i in range(r_loop):

        cv = Kfold()
        
        seed += 1