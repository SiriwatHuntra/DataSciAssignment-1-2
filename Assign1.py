import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


#Import data
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
Y = data['class']


i = 0
seed = 1
split_number = 10

bayes_model = GaussianNB()
knn_model = KNeighborsClassifier(n_neighbors=1)

acc_bayes_list = []
acc_knn_list = []


for i in range(30):
    seed += 1

    #10 fold train
    cv = KFold(n_splits= split_number, random_state= seed, shuffle= True)
        
    #accuracy of bayes
    bayes_acc = cross_val_score(bayes_model, X, Y, cv=cv)
    acc_bayes_list.append(bayes_acc)

    #accuracy of bayes
    knn_acc = cross_val_score(knn_model, X, Y, cv=cv)
    acc_knn_list.append(knn_acc)

    #Rotage seed, add i
    i+=1
    
avr_bayes = np.mean(acc_bayes_list)
avr_knn = np.mean(acc_knn_list)

#print output
print(f"Accuracy of Bayes: {avr_bayes:.4f}")
print(f"Accuracy of KNN: {avr_knn:.4f}")

#Put data into excel
out_df = {"AVR_Bayes": acc_bayes_list,
              "AVR_KNN": acc_knn_list,}
df = pd.DataFrame(out_df)
df.to_csv("AccuracyOfMoodel.csv", index = True)
print("################")
 
