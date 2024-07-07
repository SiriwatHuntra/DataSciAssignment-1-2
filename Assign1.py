import pandas as pd
import numpy as np

#Import data
def read_data_from(data_link):
    data =pd.read_csv(data_link)
    #print(data.columns) 
    X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
    Y = data['class']

