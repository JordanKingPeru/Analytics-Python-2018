# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:56:44 2018

@author: Usuario
"""
"""
-----------------------------------
ARREGLOS
    Conjunto de datos dispuestos en filas y columnas
-----------------------------------
"""
"1. Arreglos lineales
notas=[20,12,14,15,18,5]    #Lista
a = (1, 2, 3)               #Tuplas (inmutables)
b = {3, 1, 2, 1}            #Sets (conjunto: no se repite los elementos)
b
c = {'x': 1, 'y': 2, 'z': 3}#Diccionarios: "Key:Value"

len(notas)
notas.append(17)
notas.extend(a)
notas.index(12) # 1, los indices comienzan en CERO
notas.insert(3,13)
notas.count(20)
notas.sort() #notas.reverse() , notas.sort(reverse=True)

"recorrido de un array

for n in notas: print(n)
for k, v in c.items(): print(k + "=>" + str(v))

"2. Matrices
A = [
     ['Roy',80,75,85,90,95],
     ['John',75,80,75,85,100],
     ['Dave',80,80,80,90,95]
     ]
print(A[0])
print(A[0][1])

"""
Manejo mas sofisticado de las Estructuras de Datos
"""
import numpy as np
notas=np.array([20,12,14,15,18,5])
print(notas)
type(notas)     #numpy.ndarray
notas.dtype     #Tipo de Array
notas.size      #Tamaño del arreglo
notas.sum()
notas.max()
notas.min()
notas.mean()

notas[3]        #Leer un dato
notas[[2,4,5]]  #Leer varios Datos

notas[1]=17    #Reemplazar
c = np.array([1, 2, 3], dtype=float)
c.dtype

"Recorrido de un array
i=0
for x in notas:
  i=i+1
  print("Elemento : " + str(i) + ": " + str(x))

"Autogeneración
a=np.repeat(4,3)        #Repeticiones 4 4 4
b=np.arange(2,15,3)     #Secuencial 2  5  8 11 14

" Matrices
A=np.array([10,11,12,13,14,15]).reshape(3, 2)   #Por fila

B=np.array([10,11,12,13,14,15]).reshape(3, 2, order='F')    #Por Columna

B
B[2,0]      #Leer 1 dato (3ra fila,1ra columna)
B[1,:]      #Leer toda una fila
B[:,0]      #Leer toda una columna
B[[1,2],1]  #Leer la Fila 1 y 2 de la columna 1 

A.shape
A.transpose()















