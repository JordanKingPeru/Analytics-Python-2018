# -*- coding: utf-8 -*-
"""
Editor de Spyder
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

"""
#Ejemplo
matplotlib.rcParams.update({'font.size': 21})
ax = plt.gca()

ax2 = ax.twinx()
for i in range(10):
    ax.bar(i, np.random.randint(1000))

plt.ylabel('Datos')
plt.savefig("Ejemplo1.jpg")

"""

#Importando la data
df=pd.read_csv("D:/BDA_VIII/DataSets/tortuga.txt", delimiter="\t")
df.head()

v=pd.DataFrame({"variable": df.columns.values})
#Seteo de Variables Categoricas
df.Sexo=df.Sexo.astype('category')

t=pd.DataFrame({"tipo": df.dtypes.values})
meta = pd.concat([v, t], axis=1)

#GENERARCION DE GRAFICOS
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(10,5))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        fa=df[v].value_counts() 
        fr=fa/len(df[v]) 
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
        plt.savefig(v+".jpg")
    else:
        #Histograma
        plt.subplot(1,2,1)
        plt.hist(df[v])
        plt.title(v)
        #Boxplot
        plt.subplot(1,2,2)
        plt.boxplot(df[v])
        plt.title(v)
        #Guardar
        plt.savefig(v+".jpg")
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

del(x)
del(x1)





