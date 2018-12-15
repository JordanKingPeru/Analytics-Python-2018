# -*- coding: utf-8 -*-
#"Importando librerias
import numpy as np              #Matemática
import pandas as pd             #Tablas o DataFrames
import scipy.stats as sc        #Calculos estadisticos
import matplotlib.pyplot as plt #Graficos
import matplotlib

#"Importando la data
df=pd.read_csv("D:/Jose/Cursos/Big Data/Python/CASO/DataSet/Churn_Modelling_Sample.txt", delimiter="\t")
df.head()

ruta_univariado="D:/Jose/Cursos/Big Data/Python/CASO/Graficos_univariado"
ruta_bivariado="D:/Jose/Cursos/Big Data/Python/CASO/Graficos_bivariado"
ruta_resumen="D:/Jose/Cursos/Big Data/Python/CASO/resumen"

"""
TRATAMIENTO DE DATOS
"""

ds=df
P95=np.percentile(ds['Age'],q=95) # P95 :60
P95
"""
X=ds.iloc[:,3].values
X[:,0]
i=0
for v in X:
    if(v>60): X[i]=60
    i+=1
ds["Age"]=pd.DataFrame(X)
"""
ds.loc[ds["Age"]>60,"Age"]=60

plt.boxplot(ds["Age"])

"""---------------------------------------------------
Generando los inputs para el Analisis UNIVARIADO
---------------------------------------------------       
"""
ds1=df[
       ['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited']
]
       
v=pd.DataFrame({"variable": ds1.columns.values})
#Seteo de Variables Categoricas
ds1.Geography=ds1.Geography.astype('category')
ds1.Gender=ds1.Gender.astype('category')
ds1.HasCrCard=ds1.HasCrCard.astype('category')
ds1.IsActiveMember=ds1.IsActiveMember.astype('category')
ds1.Exited=ds1.Exited.astype('category')

df.dtypes
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
        plt.savefig(ruta_univariado + "/" + v + "_UNI.jpg")
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
        plt.savefig(ruta_univariado + "/" + v + "_UNI.jpg")
        #plt.show()
        
#GENERACION DE INDICADORES 
import scipy.stats as sc 
df=ds1
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
        P25=np.percentile(df[v],q=25)
        P75=np.percentile(df[v],q=75)
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
x1.to_csv(ruta_resumen + "/resumen.csv")
    
"""---------------------------------------------------
Generando los inputs para el Analisis BIVARIADO
---------------------------------------------------       
"""
#target:
y="Exited"
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(11,6))
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
        plt.savefig(ruta_bivariado + "/" + v + "_UNI.jpg")
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
        plt.savefig(ruta_bivariado + "/" + v + "_UNI.jpg")
        plt.show()