# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 17:10:17 2018

@author: Usuario
"""

"""
##########################################
## Variables y Operadores
#########################################
"""
X=3
Y=5

print(X) #mostrando la variable 
type(X) #tipo de dato

# Concatenar 
z="Hola "*3 
print(z + " " + "Mundo.....")

# Re definiendo tipos 
x=3.5 
type(x)
x=int(x) #cast
"Operadores Artimeticos (+,-,*,/,%)
 5%2
 "hhmmss ----> mm ss (102315)
 102315%100 #ss 
 int(((102315-102315%100)/100)%100) #mm
"""
-------------------------------
INSTRUCCIONES DE CONTROL
-------------------------------
"""
"CONDICIONALES
x=7
if (x%2==0) :
    print(str(x)+" Es par")
else : 
    print(str(x)+" Es impar")

"REPETITIVAS
"1. Nro de repeticiones conocido
n=0
for i in range(1,10,2):
    print(i)
    n+=1 #n=n+1 : Contador
    
"1. Nro de repeticiones desconocido
anio=2001
while anio<2012:
    print("Informe del año :" + str(anio))
    anio+=2 # anio=anio+2 : Acumulador
"""
-------------------------------
FUNCIONES
-------------------------------
"""
def max2numeros(a,b):
    m=(abs(a+b)+abs(a-b))/2
    return(m)
    
x=max2numeros(8,5)

def factorial(n):
    if (n>1):
        return(n*factorial(n-1))
    else:
        return(1)

factorial(4)





















