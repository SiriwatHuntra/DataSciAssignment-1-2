import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


#Import data
def read_data_from(data_link):
    data =pd.read_csv(data_link, header= None)
    print(data.columns)
    data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

    X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
    Y = data['class']
    return X, Y

def test_model(r_loop):
    seed = 1
    split_number = 10
    bayes_model = GaussianNB()
    knn_model = KNeighborsClassifier(n_neighbors=1)
    acc_bayes = []
    acc_knn = []

    knn_model = KNeighborsClassifier(n_neighbors=1)
    X, Y = read_data_from('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

    for i in range(r_loop):
        
        #10 fold train
        cv = KFold(n_splits= split_number, random_state= seed, shuffle= True)
        
        #accuracy of bayes
        bayes_acc = cross_val_score(bayes_model, X, Y, cv=cv)
        acc_bayes.append(bayes_acc)

        #accuracy of bayes
        knn_acc = cross_val_score(knn_model, X, Y, cv=cv)
        acc_knn.append(knn_acc)

        #Rotage seed, add i
        seed += 1
        i+=1
    
    avr_bayes = np.mean(acc_bayes)
    avr_knn = np.mean(acc_knn)

    #print output
    print(f"Accuracy of Bayes: {avr_bayes:.2f}")
    print(f"Accuracy of KNN: {avr_knn:.2f}")

    #Put data into excel
    out_df = {"AVR_Bayes": avr_bayes,
              "AVR_KNN": avr_knn}
    pd.to_csv("AccuracyOfMoodel.csv", index = True)
    print("################")
 
test_model(30)