# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Importando la Data de prueba
ds=pd.read_csv("D:/BDA_VIII/DataSets/Churn_Modelling.csv")
ds.head()
ds.columns.values
ds1=ds[
       ['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited']
]
"""---------------------------------------------------
Generando los inputs para el Analisis UNIVARIADO
---------------------------------------------------       
"""
v=pd.DataFrame({"variable": ds1.columns.values})
#Seteo de Variables Categoricas
ds1.Geography=ds1.Geography.astype('category')
ds1.Gender=ds1.Gender.astype('category')
ds1.HasCrCard=ds1.HasCrCard.astype('category')
ds1.IsActiveMember=ds1.IsActiveMember.astype('category')
ds1.Exited=ds1.Exited.astype('category')

ds1.dtypes
t=pd.DataFrame({"tipo": ds1.dtypes.values})
meta = pd.concat([v, t], axis=1)

#GENERACION DE GRAFICOS
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(10,5))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        fa=ds1[v].value_counts() 
        fr=fa/len(ds1[v]) 
        #Barras
        plt.subplot(1,2,1)
        plt.bar(fa.index,fa)
        plt.xticks(fa.index)
        plt.title(v)
        #Pie
        plt.subplot(1,2,2)
        plt.pie(fr,autopct='%1.1f%%', shadow=True, startangle=90)
        plt.legend(fr.index,loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
        plt.title(v)
        #Guardar
        plt.savefig("univariado1/"+ v + ".jpg")
    else:
        #Histograma
        plt.subplot(1,2,1)
        plt.hist(ds1[v])
        plt.title(v)
        #Boxplot
        plt.subplot(1,2,2)
        plt.boxplot(ds1[v])
        plt.title(v)
        #Guardar
        plt.savefig("univariado1/"+ v + ".jpg")
    plt.show()
        
#GENERACION DE INDICADORES 
import scipy.stats as sc 

for i in range(len(meta)) :
    v=meta.iloc[i].variable 
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        x=pd.DataFrame({"var":v,"mean": ".",
                        "median": ".",
                        "mode": ".",
                        "min": ".",
                        "max": ".",
                        "sd": ".",
                        "cv": ".",
                        "k": ".",
                        "Q1": ".",
                        "Q3": ".",
                        "Nmiss": "."
                        },index=[i])
    else:
        P25=np.percentile(df["Altura"],q=25)
        P75=np.percentile(df["Altura"],q=75)
        IQR=P75-P25
        liV=P25-1.5*IQR # Mimimo Viable
        lsV=P75+1.5*IQR # Maximo Viable
        x=pd.DataFrame({"var":v,"mean": df[v].mean(),
                        "median": df[v].median(),
                        "mode": df[v].mode(),
                        "min": df[v].min(),
                        "max": df[v].max(),
                        "sd": df[v].std(),
                        "cv": df[v].std()/df[v].mean(),
                        "k": sc.kurtosis(df[v]),
                        "Q1": np.percentile(df[v],q=25),
                        "Q3": np.percentile(df[v],q=75),
                        "Nmiss": df[v].isnull().sum()/len(df)
                        }, index=[i])
    if(i==0):
        x1=x
    else:
       x1=pd.concat([x1, x])  # x1.append(x)
    print(i)
 