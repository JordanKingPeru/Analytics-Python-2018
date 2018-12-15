# -*- coding: utf-8 -*-
"Importando librerias
import numpy as np              #Matemática
import pandas as pd             #Tablas o DataFrames
import scipy.stats as sc        #Calculos estadisticos
import matplotlib.pyplot as plt #Graficos

"Importando la data
df=pd.read_csv("D:/BDA_VIII/DataSets/tortuga.txt", delimiter="\t")
df.head()

"1.Variables Discretas
fa=df["Sexo"].value_counts()     #Freq Abs
fr=fa/len(df)                    #Freq Rel
faA=fa.cumsum()                  #Freq Acumulada
"  Materialidad
plt.bar(fa.index,fa)
plt.ylabel("Nro de Tortugas")
plt.title("Distribución segun Sexo")
plt.xticks(fa.index)
" Cobertura
plt.pie(fr,autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Cobertura segun Sexo")
plt.legend(fr.index)

"2.Variables Continuas
 " Distribución
df['Altura'].hist()

plt.hist(df['Altura'])
plt.ylabel("Cantidad")
plt.title("Distribución segun Altura")

plt.boxplot(df['Altura'])
plt.ylabel("Cantidad")
plt.title("Diagrama de Caja variable Altura")
 " Indicadores Estadisticos
    "1. Medidas de Tendencia Central
df['Altura'].mean()             # Promedio Artimetico de los datos
df['Altura'].median()           # Punto medio de los datos
most_freq=df['Altura'].mode()   # El dato que mas repite
most_freq[0]
    " 2. Medidas de Dispersión
df['Altura'].max()
df['Altura'].min()
rango=df['Altura'].max()-df['Altura'].min()
rango # Rango de variación
df['Altura'].var()
df['Altura'].std()

ls=df['Altura'].mean()+df['Altura'].std()
li=df['Altura'].mean()-df['Altura'].std()
ls-li # Limites de control

df['Altura'].std()/df['Altura'].mean()
df['Longitud'].std()/df['Longitud'].mean()
    " 3. Medidas de Forma
sc.kurtosis(df['Altura'])  #-0.37 la variable esta concentra en valores menores
    
    " 4. Medidas de Posición
np.percentile(df["Altura"],q=[25,50,75])
np.percentile(df["Altura"],q=1)  #P1
np.percentile(df["Altura"],q=99) #P99
np.percentile(df["Altura"],q=range(0,100,10)) #Deciles
    
    " 5. Prametros para el tramiento de Datos    
NaN=df['Altura'].isnull().sum()
PctNull=NaN/len(df) # Porcentaje de Nulos

P25=np.percentile(df["Altura"],q=25)
P75=np.percentile(df["Altura"],q=75)
IQR=P75-P25
liV=P25-1.5*IQR # Mimimo Viable
lsV=P75+1.5*IQR # Maximo Viable

len(df[['Altura']][(df.Altura<liV)])
len(df[['Altura']][(df.Altura>lsV)])



