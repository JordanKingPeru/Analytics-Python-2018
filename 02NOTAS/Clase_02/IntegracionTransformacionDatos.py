# -*- coding: utf-8 -*-

" Importing the libraries
import numpy as np
import pandas as pd

" Importacion Local en formato neutro (txt,csv,....plano)
location="D:/BDA_VIII/DataSets/tortuga.txt"
df=pd.read_csv(location, delimiter="\t")
df
df.head()           #Visualizar los primeros registros
df.columns.values   #Lista de variables/atributos
df.describe()       #Auditoria de la tabla)

"Importacion Local en formato no neutro (sav,sas,..)
import savReaderWriter as spss  #pip install savReaderWriter

location="D:/BDA_VIII/DataSets/vinos.sav"
with spss.SavReaderNp (location) as reader:
    records = reader.all()

ds = pd.DataFrame(records)
ds.head()
"""
------------------
SELECCION Y FILTROS
------------------
"""
"   SELECCION (SELECT)
df.iloc[:, [2]].values          # 1 Variable
df.Sexo
df.iloc[:, [0, 3]].values       # ds1[c(1,4)] Varias cols
df[['Ancho','Sexo','Altura']]
df.filter(items=['Longitud', 'Sexo'])
df.filter(regex='o$', axis=1)   # select columns by regular expression
"   FILTROS (WHERE)

df.ix[1]                    # 1 Observación por indice
df.ix[[4,11,33]]            # n Observaciones por indice

df[df.Ancho >=110]       # filtro lógico
df[~(((df.Sexo==1) & (df.Altura<60)) | ((df.Sexo==0) & (df.Ancho<100)))]

"SELECT Ancho, Sexo, Altura FROM ds1 WHERE NOT (Sexo==1 & Altura<60)
df1=df[['Ancho','Sexo','Altura']][~((df.Sexo==1) & (df.Altura<60))]

"""
------------------
INTEGRACION
------------------
"""
"   JOINS
caller = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                        'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})

other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                       'B': ['B0', 'B1', 'B2']})

x=caller.join(other, lsuffix='_caller', rsuffix='_other')
y=caller.join(other.set_index('key'), on='key')