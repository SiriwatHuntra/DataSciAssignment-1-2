# Import Lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing

#Load dataset
cal_housing = fetch_california_housing()
df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)

#add target val to dataframe
df['target'] = cal_housing.target

data = df.sample(n=10000) #Try 400, 4000, 10000, all
#data = df


#data visual
df.to_csv("Cal_house.csv", index = True)

#Setup dataframe
X = data[["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]]
Y = data["target"]

#load model
baseModel = LinearRegression()
cross_val = KFold(n_splits=10, random_state=0, shuffle=True)

#check kfold score
kf_score = cross_val_score(baseModel, X, Y, cv=cross_val, scoring='r2')
mean = np.mean(kf_score)
print(f'Score of KFold: {kf_score}')
print(f'ACC: {mean:.4f}')

#corRelate
cor_relation = data.corr()
plt.matshow(cor_relation)
plt.colorbar()
plt.show()

data.boxplot()
plt.show()