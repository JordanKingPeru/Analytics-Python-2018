# -*- coding: utf-8 -*-

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('ds_construccion.csv')
X = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 4].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 0)
# Feature Scaling (Regresion ,  EstimatedSalary)
X_train=X_train.astype(float)
X_train[:,2].mean() #68147.61904761905
X_train[:,2].std()  #34703.260709260096
X_train[:,2]=(X_train[:,2]-68147.61904761905)/34703.260709260096

X_train[:,1].mean() #38.02857142857143
X_train[:,1].std()  #10.21013235570864
X_train[:,1]=(X_train[:,1]-38.02857142857143)/10.21013235570864

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

################################################################
# Predicting the Test set results
X_test=X_test.astype(float)
X_test[:,2]=(X_test[:,2]-68147.61904761905)/34703.260709260096
X_test[:,1]=(X_test[:,1]-38.02857142857143)/10.21013235570864

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, roc_auc_score
cm = confusion_matrix(y_test, y_pred)

#Partida de nacimiento
ROC=roc_auc_score(y_test, y_pred)
Gini=2*ROC-1 #0.6336405529953917 : 63% gini de Construcción
###############################################################
#SEGUIMIENTO
###############################################################
dataset = pd.read_csv('ds_seguimiento.csv')
dataset.describe()

#Evolucion de la prediccion
Mes = []
Gini = []
for i in range(6):
    ds_s_201901=dataset[dataset.Codmes==201901+i]
    X_obs = ds_s_201901.iloc[:, [1,2,3]].values
    y_obs = ds_s_201901.iloc[:, 4].values
    
    X_obs=X_obs.astype(float)
    X_obs[:,2]=(X_obs[:,2]-68147.61904761905)/34703.260709260096
    X_obs[:,1]=(X_obs[:,1]-38.02857142857143)/10.21013235570864
    
    y_pred = classifier.predict(X_obs)
    
    ROC=roc_auc_score(y_obs, y_pred)
    Gini.append(2*ROC-1) #0.9166666666666665 : 91%
    Mes.append(201900+i)

EvolGini=pd.DataFrame({'codmes': Mes,
                       'Gini':Gini})

plt.scatter(EvolGini.codmes,EvolGini.Gini)    
plt.axhline(y=0.9166, color='r', linestyle='-')

#Evolucion de la estabilidad
dataset_c = pd.read_csv('ds_construccion.csv')
dataset_s = pd.read_csv('ds_seguimiento.csv')

Mes = []
PSI = []
for i in range(12):
    dataset_s_201901=dataset_s[dataset.Codmes==201901+i]
    #   Gender
    feq_c=dataset_c["Gender"].value_counts()/len(dataset_c) # ref
    feq_s=dataset_s_201901["Gender"].value_counts()/len(dataset_s_201901) # obs
    #PSI=sum((feq_s-feq_c)*np.log(feq_s/feq_c)) #0.2311811188277748
     #       como 0.1< PSI <0.25 :  Se presentan dudas sobre la estabilidad 
     #       seguir moniroreando
    Mes.append(201900+i)
    PSI.append(sum((feq_s-feq_c)*np.log(feq_s/feq_c)))

EvolPSI=pd.DataFrame({'codmes': Mes,
                       'PSI':PSI})

plt.scatter(EvolPSI.codmes,EvolPSI.PSI)    
plt.axhline(y=0.1, color='r', linestyle='-')
plt.axhline(y=0.25, color='r', linestyle='-')
plt.axhline(y=1.0, color='b', linestyle='-')


#######################################################
#Age
#######################################################
Mes = []
PSI = []
for i in range(12):
    dataset_s_201901=dataset_s[dataset.Codmes==201901+i]
    #   Gender
    feq_c=dataset_c["Ten"].value_counts()/len(dataset_c) # ref
    feq_s=dataset_s_201901["Age"].value_counts()/len(dataset_s_201901) # obs
    #PSI=sum((feq_s-feq_c)*np.log(feq_s/feq_c)) #0.2311811188277748
     #       como 0.1< PSI <0.25 :  Se presentan dudas sobre la estabilidad 
     #       seguir moniroreando
    Mes.append(201900+i)
    PSI.append(sum((feq_s-feq_c)*np.log(feq_s/feq_c)))

EvolPSI=pd.DataFrame({'codmes': Mes,
                       'PSI':PSI})

plt.scatter(EvolPSI.codmes,EvolPSI.PSI)    
plt.axhline(y=0.1, color='r', linestyle='-')
plt.axhline(y=0.25, color='r', linestyle='-')
plt.axhline(y=1.0, color='b', linestyle='-')




