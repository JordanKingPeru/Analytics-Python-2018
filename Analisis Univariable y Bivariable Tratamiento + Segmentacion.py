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
ruta_tratamiento="D:/Jose/Cursos/Big Data/Python/CASO/Tratamiento"
ruta_paises_altos="D:/Jose/Cursos/Big Data/Python/CASO/Paises_altos"
ruta_paises_bajos="D:/Jose/Cursos/Big Data/Python/CASO/Paises_bajos"


"""---------------------------------------------------
Generando los inputs para el Analisis UNIVARIADO
------------------------------------------------------       
"""
ds1=df[
       ['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited']
]
       
v=pd.DataFrame({"variable": ds1.columns.values})

#Seteo de Variables Categoricas

## Me salie este error 
#C:\Users\usuario\Anaconda3\lib\site-packages\pandas\core\generic.py:4405: SettingWithCopyWarning: 
#A value is trying to be set on a copy of a slice from a DataFrame.
#Try using .loc[row_indexer,col_indexer] = value instead
#lo cambie por  str( ds1.dtypes['Geography'] ) == 'category'

ds1.Geography=ds1.Geography.astype('category')
ds1.Gender=ds1.Gender.astype('category')
ds1.HasCrCard=ds1.HasCrCard.astype('category')
ds1.IsActiveMember=ds1.IsActiveMember.astype('category')
ds1.Exited=ds1.Exited.astype('category')


str( ds1.dtypes['Geography'] )== 'category'
str( ds1.dtypes['Gender'] ) == 'category'
str( ds1.dtypes['HasCrCard'] ) == 'category'
str( ds1.dtypes['IsActiveMember'] ) == 'category'
str( ds1.dtypes['Exited'] ) == 'category'

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
        

"""---------------------------------------------------
TRATAMIENTO DE DATOS
------------------------------------------------------
"""

#Segmento PA
    #Age outlier
    #Balance outlier
    #CreditScore outlier
    #NumOfProducts  Categorizanción


np.percentile(ds1['Age'],q=97)
np.percentile(ds1['Balance'],q=99)
np.percentile(ds1['Balance'],q=1)
np.percentile(ds1['CreditScore'],q=2)

ds1.loc[ds1["Age"]>=64,"Age"]=64
ds1.loc[ds1["Balance"]>=185160.94680000003,"Balance"]=185160.94680000003
ds1.loc[ds1["Balance"]<=53102.854400000004,"Balance"]=53102.854400000004
ds1.loc[ds1["CreditScore"]<=453,"CreditScore"]=453
ds1.loc[ds1["NumOfProducts"]>1,"NumOfProducts"]=0 # 1: tiene un producto , 0: tiene mas de 1 producto
ds1.NumOfProducts=ds1.NumOfProducts.astype('category')


plt.boxplot(ds1["Age"])
plt.boxplot(ds1["Balance"])
plt.boxplot(ds1["CreditScore"])
fa=ds1["NumOfProducts"].value_counts() 
plt.bar(fa.index,fa)

"""---------------------------------------------------
SEGMENTACION POR GEOGRAFIA
------------------------------------------------------
"""

ds1_PB=ds1[ds1["Geography"]!="Germany"]
ds1_PA=ds1[ds1["Geography"]=="Germany"]



"""---------------------------------------------------
Generando los inputs para el Analisis UNIVARIADO
PAISES BAJOS
---------------------------------------------------       
"""
dsb=ds1_PB
dsb.head()

v=pd.DataFrame({"variable": dsb.columns.values})
#Seteo de Variables Categoricas
dsb.Geography=dsb.Geography.astype('category')
dsb.Gender=dsb.Gender.astype('category')
dsb.HasCrCard=dsb.HasCrCard.astype('category')
dsb.IsActiveMember=dsb.IsActiveMember.astype('category')
dsb.NumOfProducts=dsb.NumOfProducts.astype('category')
dsb.Exited=dsb.Exited.astype('category')

dsb.dtypes
t=pd.DataFrame({"tipo": dsb.dtypes.values})
meta = pd.concat([v, t], axis=1)

#GENERACION DE GRAFICOS
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(10,5))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        fa=dsb[v].value_counts() 
        fr=fa/len(dsb[v]) 
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
        plt.savefig(ruta_paises_bajos + "/" + v + "_UNI.jpg")
    else:
        #Histograma
        plt.subplot(1,2,1)
        plt.hist(dsb[v])
        plt.title(v)
        #Boxplot
        plt.subplot(1,2,2)
        plt.boxplot(dsb[v])
        plt.title(v)
        #Guardar
        plt.savefig(ruta_paises_bajos + "/" + v + "_UNI.jpg")
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
        P25=np.percentile(dsb[v],q=25)
        P75=np.percentile(dsb[v],q=75)
        IQR=P75-P25
        liV=P25-1.5*IQR # Mimimo Viable
        lsV=P75+1.5*IQR # Maximo Viable
        x=pd.DataFrame({"var":v,"mean": dsb[v].mean(),
                        "median": dsb[v].median(),
                        "mode": dsb[v].mode(),
                        "min": dsb[v].min(),
                        "max": dsb[v].max(),
                        "sd": dsb[v].std(),
                        "cv": dsb[v].std()/dsb[v].mean(),
                        "k": sc.kurtosis(dsb[v]),
                        "Q1": np.percentile(dsb[v],q=25),
                        "Q3": np.percentile(dsb[v],q=75),
                        "Nmiss": dsb[v].isnull().sum()/len(dsb)
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
        g=dsb.groupby([ds1[y],v]).size().unstack(0)
        tf= g[1]/(g[0]+g[1])
        c1 = g[0]
        c2 = g[1]
        width = 0.9       # the width of the bars: can also be len(x) sequence
        
        p1 = plt.bar(g.index, c1, width)
        p2 = plt.bar(g.index, c2, width,
                     bottom=c1)
        
        plt.ylabel('Freq')
        plt.title('Bivariado: ' + v + " vs " + y)
        plt.xticks(g.index)
        plt.legend((p1[0], p2[0]), ('0', '1'),loc='lower left',bbox_to_anchor=(1, 1))
        
        plt.twinx().plot(tf.values,linestyle='-', linewidth=2.0,color='red')
        plt.ylabel('Ratio Fuga')
        #Guardar
        plt.savefig(ruta_paises_bajos + "/"+ v + "_BIVAR.jpg")
    else:
        d=pd.qcut(dsb[v], 10, duplicates='drop',labels=False)     
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
        plt.savefig(ruta_paises_bajos + "/"+ v + "_BIVAR.jpg")
    plt.show()


"""---------------------------------------------------
Generando los inputs para el Analisis UNIVARIADO
PAISES ALTOS
---------------------------------------------------       
"""
dsa=ds1_PA
dsa.head()

v=pd.DataFrame({"variable": dsa.columns.values})
#Seteo de Variables Categoricas
dsa.Geography=dsa.Geography.astype('category')
dsa.Gender=dsa.Gender.astype('category')
dsa.HasCrCard=dsa.HasCrCard.astype('category')
dsa.IsActiveMember=dsa.IsActiveMember.astype('category')
dsa.NumOfProducts=dsa.NumOfProducts.astype('category')
dsa.Exited=dsa.Exited.astype('category')

dsa.dtypes
t=pd.DataFrame({"tipo": dsa.dtypes.values})
meta = pd.concat([v, t], axis=1)

#GENERACION DE GRAFICOS
matplotlib.rcParams.update({'font.size': 16})
for i in range(len(meta)) :
    plt.figure(figsize=(10,5))
    v=meta.iloc[i].variable #print(meta.iloc[i].variable)
    t=meta.iloc[i].tipo
    if (t.__class__.__name__=="CategoricalDtype"):
        fa=dsa[v].value_counts() 
        fr=fa/len(dsa[v]) 
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
        plt.savefig(ruta_paises_altos + "/" + v + "_UNI.jpg")
    else:
        #Histograma
        plt.subplot(1,2,1)
        plt.hist(dsa[v])
        plt.title(v)
        #Boxplot
        plt.subplot(1,2,2)
        plt.boxplot(dsa[v])
        plt.title(v)
        #Guardar
        plt.savefig(ruta_paises_altos + "/" + v + "_UNI.jpg")
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
        P25=np.percentile(dsa[v],q=25)
        P75=np.percentile(dsa[v],q=75)
        IQR=P75-P25
        liV=P25-1.5*IQR # Mimimo Viable
        lsV=P75+1.5*IQR # Maximo Viable
        x=pd.DataFrame({"var":v,"mean": dsa[v].mean(),
                        "median": dsa[v].median(),
                        "mode": dsa[v].mode(),
                        "min": dsa[v].min(),
                        "max": dsa[v].max(),
                        "sd": dsa[v].std(),
                        "cv": dsa[v].std()/dsa[v].mean(),
                        "k": sc.kurtosis(dsa[v]),
                        "Q1": np.percentile(dsa[v],q=25),
                        "Q3": np.percentile(dsa[v],q=75),
                        "Nmiss": dsa[v].isnull().sum()/len(dsa)
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
        g=dsa.groupby([ds1[y],v]).size().unstack(0)
        tf= g[1]/(g[0]+g[1])
        c1 = g[0]
        c2 = g[1]
        width = 0.9       # the width of the bars: can also be len(x) sequence
        
        p1 = plt.bar(g.index, c1, width)
        p2 = plt.bar(g.index, c2, width,
                     bottom=c1)
        
        plt.ylabel('Freq')
        plt.title('Bivariado: ' + v + " vs " + y)
        plt.xticks(g.index)
        plt.legend((p1[0], p2[0]), ('0', '1'),loc='lower left',bbox_to_anchor=(1, 1))
        
        plt.twinx().plot(tf.values,linestyle='-', linewidth=2.0,color='red')
        plt.ylabel('Ratio Fuga')
        #Guardar
        plt.savefig(ruta_paises_altos + "/"+ v + "_BIVAR.jpg")
    else:
        d=pd.qcut(dsa[v], 10, duplicates='drop',labels=False)     
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
        plt.savefig(ruta_paises_altos + "/"+ v + "_BIVAR.jpg")
    plt.show()