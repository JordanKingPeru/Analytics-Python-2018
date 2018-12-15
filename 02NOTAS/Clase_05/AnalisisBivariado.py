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
"""      
Para Modelos de Clasificación 
"""
# se raliza un versus entre cada Variable predictora y el target
y="Exited" # tarjet (1: Fuga y 0: No fuga)

# 1. variable preditora es discreta:
x="Gender"
g=ds1.groupby([ds1[y],x]).size().unstack(0)
tasaFuga=g[1]/(g[0]+g[1])
    #Grafico
width=0.9
plt.title('Analisis Bivariado(' + x + " vs " + y + ")")
    #Eje principal
p1 = plt.bar(g.index, g[0], width)
p2 = plt.bar(g.index, g[1], width,bottom=g[0])
plt.ylabel('Freq')
plt.xlabel(x)
plt.legend((p1[0], p2[0]), ('0', '1'),bbox_to_anchor=(1, 1.2))
    #Eje secundario
plt.twinx().plot(tasaFuga.values, linestyle="-",color="red")
plt.ylabel('Tasa de Fuga')



# 2. variable preditora es continua:
x="Balance"
d=pd.qcut(ds1[x], 10,duplicates='drop', labels=False) 
g=ds1.groupby([ds1[y],d]).size().unstack(0)
tasaFuga=g[1]/(g[0]+g[1])
    #Grafico
width=0.9
plt.title('Analisis Bivariado(' + x + " vs " + y + ")")
    #Eje principal
p1 = plt.bar(g.index, g[0], width)
p2 = plt.bar(g.index, g[1], width,bottom=g[0])
plt.ylabel('Freq')
plt.xlabel(x)
plt.legend((p1[0], p2[0]), ('0', '1'),bbox_to_anchor=(1, 1.2))
    #Eje secundario
plt.twinx().plot(tasaFuga.values, linestyle="-",color="red")
plt.ylabel('Tasa de Fuga')

"""
AUTOMATIZACION DEL CODIGO
"""
#GENERACION DE GRAFICOS
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
#target:
y="Exited"
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(10,5))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if v==y: break
    print(v)
    if (t.__class__.__name__=="CategoricalDtype"):        
        g=ds1.groupby([ds1[y],v]).size().unstack(0)
        tf= g[1]/(g[0]+g[1])
        c1 = g[0]
        c2 = g[1]
        width = 0.9       # the width of the bars: can also be len(x) sequence
        
        p1 = plt.bar(g.index, c1, width)
        p2 = plt.bar(g.index, c2, width,
                     bottom=c1)
        
        plt.ylabel('Freq')
        plt.title('Bivariado')
        plt.xticks(g.index)
        plt.legend((p1[0], p2[0]), ('0', '1'),loc='lower left',bbox_to_anchor=(1, 1))
        
        plt.twinx().plot(tf.values,linestyle='-', linewidth=2.0,color='red')
        plt.ylabel('Ratio Fuga')
        #Guardar
        plt.savefig("Clase_05/Bivariado1/"+ v + ".jpg")
    else:
        d=pd.qcut(ds1[v], 10, duplicates='drop',labels=False)     
        g=ds1.groupby(['Exited', d]).size().unstack(0)   
        N = len(g)
        menMeans = g[0]
        womenMeans = g[1]
        tf= g[1]/(g[0]+g[1])
        ind = np.arange(N)    # the x locations for the groups

        width = 0.9       # the width of the bars: can also be len(x) sequence        
        p1 = plt.bar(ind, menMeans, width)
        p2 = plt.bar(ind, womenMeans, width,
                     bottom=menMeans)
        
        plt.ylabel('Freq')
        plt.xlabel("Deciles " + v)
        plt.title('Bivariado: ' + v + " vs " + y)
        plt.xticks(ind, np.arange(1,10,1))
        plt.legend((p1[0], p2[0]), ('0', '1'),loc='lower left',bbox_to_anchor=(1, 1))
        
        plt.twinx().plot(tf.values,linestyle='-', linewidth=2.0,color='red')
        plt.ylabel('Ratio Fuga')
        #Guardar
        plt.savefig("Clase_05/Bivariado1/"+ v + ".jpg")
    plt.show()







