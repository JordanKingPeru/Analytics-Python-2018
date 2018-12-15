# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:03:39 2018

@author: Usuario
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando la Data de prueba
ds=pd.read_csv("D:/BDA_VIII/DataSets/Data.csv")
ds.head()



# 1. tratamiento de missings
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
imp=imp.fit(X[:, 1:3])
X[:,1:3]=imp.transform(X[:,1:3])
ds_tratado=pd.DataFrame(X)
ds_tratado.columns=["Country","Age","Salary"]


# 2. tratamiento de cotas
ds=pd.read_csv("D:/BDA_VIII/DataSets/tortuga.txt", delimiter="\t")
ds.head()

P95=np.percentile(ds['Ancho'],q=95) # P95 :117.65
P95
X=ds.iloc[:,:-1].values
i=0
for v in X[:,1]:
    if(v>117.65): X[i,1]=117.65
    i+=1
ds_tratado=pd.DataFrame(X)
plt.boxplot(ds_tratado[1])

# 3. Codificacion de variables categoricas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lblEnc=LabelEncoder()
X=ds.iloc[:,:-1].values # Variables independientes (Predictoras)
X[:,0]=lblEnc.fit_transform(X[:,0])
X

# 4. Escalamiento de Variables
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
ds_tratado=pd.DataFrame(X)

# 5. Separacion de muestras en Train y Test
ds=pd.read_csv("D:/BDA_VIII/DataSets/tortuga.txt", delimiter="\t")
ds.head()
X = ds.iloc[:, :-1].values
y = ds.iloc[:, 3].values
from sklearn.cross_validation import train_test_split
X_train, X_test,y_train, y_test =train_test_split(X,y,test_size = 0.2,random_state = 0)














