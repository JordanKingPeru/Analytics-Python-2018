# -*- coding: utf-8 -*-
"""Jordan King Rodriguez Mallqui
Created on Thu Dec 20 23:53:53 2018

@author: Usuario
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

ds=pd.read_csv("SegmentoPaisesBajos_PB/ds1_PB.csv")

ds_cluster = ds[["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]

from sklearn.decomposition import PCA
import sklearn.cluster as clu

help(sklearn.cluster)
pca = PCA(n_components=2)
pca = pca.fit(ds_cluster)
ds_cluste_pca = pca.transform(ds_cluster)

existing_df_2d = pd.DataFrame(ds_cluste_pca)
existing_df_2d.columns = ['PC1','PC2']
existing_df_2d.head()

existing_df_2d.plot(
    kind='scatter', 
    x='PC2', y='PC1', 
    #s=existing_df_2d['country_change_scaled']*100, 
    figsize=(16,8))

ds_cluste_pca.explained_variance_
ds_cluste_pca.explained_variance_ratio_
ds_cluste_pca




clu.estimate_bandwidth(ds_cluster, quantile=0.3, n_samples=None, random_state=0, n_jobs=1)
ds_kmeans = clu.k_means(ds_cluster, 4, init='k-means++', precompute_distances='auto', n_init=10, max_iter=300, verbose=False, tol=0.0001, random_state=None, copy_x=True, n_jobs=1, algorithm='auto', return_n_iter=False)

kmeans = clu.KMeans(n_clusters=4)
kmeans = kmeans.fit(ds_cluster)
labels = kmeans.predict(ds_cluster)
centroids = kmeans.cluster_centers_

ds['PC1'] = existing_df_2d['PC1']
ds['PC2'] = existing_df_2d['PC2']
ds['cluster'] = existing_df_2d['cluster']

existing_df_2d['cluster'] = pd.Series(kmeans.labels_, index=existing_df_2d.index)

existing_df_2d.plot(
        kind='scatter',
        x='PC2',y='PC1',
        c=existing_df_2d.cluster.astype(np.float),
        cmap='viridis',
        figsize=(16,8),
        )

existing_df_2d.plot(
        kind='scatter',
        x='PC2',y='PC1',
        c=existing_df_2d.cluster.astype(np.float),
        cmap='viridis',
        figsize=(16,8),
        )


import numpy as np
import matplotlib.pyplot as plt

data=ds[['EstimatedSalary','cluster']]
data.boxplot(by='cluster',grid=False,figsize=(16,8))

ds.groupby(['cluster']).count()


df = pd.DataFrame(np.random.randn(10, 2),columns=['Col1', 'Col2'])
df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A','B', 'B', 'B', 'B', 'B'])
boxplot = df.boxplot(by='X')


# -*- coding: utf-8 -*-

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Clase_10/ds_construccion.csv')
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
dataset = pd.read_csv('Clase_10/ds_seguimiento.csv')
dataset.describe()

#Evolucion de la prediccion
ds_s_201901=dataset[dataset.Codmes==201906]
X_obs = ds_s_201901.iloc[:, [1,2,3]].values
y_obs = ds_s_201901.iloc[:, 4].values

X_obs=X_obs.astype(float)
X_obs[:,2]=(X_obs[:,2]-68147.61904761905)/34703.260709260096
X_obs[:,1]=(X_obs[:,1]-38.02857142857143)/10.21013235570864

y_pred = classifier.predict(X_obs)

ROC=roc_auc_score(y_obs, y_pred)
Gini=2*ROC-1 #0.9166666666666665 : 91%

EvolGini=pd.DataFrame({'codmes': [201901,201902,201903,201904,201905,201906],
                       'Gini':[91,88,93,99,57,48]})

#Evolucion de la estabilidad
dataset_c = pd.read_csv('Clase_10/ds_construccion.csv')
dataset_s = pd.read_csv('Clase_10/ds_seguimiento.csv')
dataset_s_201901=dataset_s[dataset.Codmes==201901]
#   Gender
feq_c=dataset_c["Gender"].value_counts()/len(dataset_c) # ref
feq_s=dataset_s_201901["Gender"].value_counts()/len(dataset_s_201901) # obs
PSI=sum((feq_s-feq_c)*np.log(feq_s/feq_c)) #0.2311811188277748
 #       como 0.1< PSI <0.25 :  Se presentan dudas sobre la estabilidad 
 #       seguir moniroreando


