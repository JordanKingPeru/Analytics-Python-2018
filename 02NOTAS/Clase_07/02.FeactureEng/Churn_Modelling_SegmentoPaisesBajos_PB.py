# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Importando la Data de prueba
ds=pd.read_csv("D:/Cursos/Big Data & Analytics/Analytics-Python-2018/SegmentoPaisesBajos_PB/ds1_PB.csv")
ds=ds.drop(["Unnamed: 0"], axis=1)
ds.head()
ds.info()

#====================================
# Analisis de Correlaciones
#====================================
ds1=ds.drop(["Geography","Exited"], axis=1)

#Matrix de correlacion
import seaborn as sns

corr=ds1.corr()

#Mapa de Calor
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#lista de correlados

lc=corr.abs().unstack().sort_values(kind="quicksort")
lc
#====================================
# Preparacion Final
#====================================
ds.info()
ds=ds.drop(["Geography"], axis=1)
ds=ds.drop(["RowNumber"], axis=1)
ds=ds.drop(["Surname"], axis=1)
ds=ds.drop(["CustomerId"], axis=1)

#Separacion de variables predictivas y target
X=ds.iloc[:,:-1].values
y=ds.iloc[:,9].values

X[:,1] # Dato Descriptivo

#Codificacion de los datos categoricos X[:,1]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le=LabelEncoder()
X[:,1]=le.fit_transform(X[:,1])
ohe=OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X=X[:,1:]
xxxx=pd.DataFrame(X)
data_feature_names=['Gender','CreditScore',  'Age', 'Tenure', 'Balance', 'NumOfProducts',
                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
# Separancion de train y test

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=0)

#==============================================
#Entrenamiento Arbol de Clasificación
#==============================================

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train,y_train)

#Acerca del modelo
print(classifier)
classifier.get_params()
importance=classifier.feature_importances_
plt.barh(data_feature_names,importance)

# Visualizando el arbol online: http://www.webgraphviz.com
from sklearn import tree
tree.export_graphviz(classifier, 
                     feature_names=data_feature_names,
                     out_file="Clase_07/02.FeactureEng/tree_PB.dot",
                     filled=True)





















