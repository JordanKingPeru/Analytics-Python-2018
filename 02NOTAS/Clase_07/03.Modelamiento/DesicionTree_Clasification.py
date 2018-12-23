# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
#=============================================
# 1. DATA PREPARATION (ON MEMORY)
#=============================================
# Importando la Data de prueba
ds=pd.read_csv("SegmentoPaisesBajos_PB/ds1_PB.csv")
ds=ds.drop(["Unnamed: 0"], axis=1)
ds.info()

ds=ds.drop(["Geography"], axis=1) # Eliminamos la variable segmentadora
ds=ds.drop(["RowNumber"], axis=1)
ds=ds.drop(["CustomerId"], axis=1)
ds=ds.drop(["Surname"], axis=1)
ds=ds.drop(["Geography"], axis=1)

#Separacion de variables predictivas del target
X=ds.iloc[:,:-1].values
y=ds.iloc[:,9].values

X[:,1] # Dato Descriptivo ['Male', 'Female',]

#Codificacion de los datos categoricos X[:,1]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le=LabelEncoder()
X[:,1]=le.fit_transform(X[:,1])
ohe=OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X=X[:,1:]
xxxx=pd.DataFrame(X)
data_feature_names=['Gender','CreditScore',  'Age', 'Tenure', 'Balance', 'NumOfProducts',
                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
# Separancion de train y test

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=0)

#==============================================
# 2. PRIMER ENTRENAMIENTO
#==============================================

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train,y_train)
# Acerca del modelo (Metricas)
classifier.get_params()
    # A. Importancia de las Variables
importance=classifier.feature_importances_
#importance=np.sort(importance)
matplotlib.rcParams.update({'font.size':15})
plt.barh(data_feature_names,importance*100)

for i in range(0,len(importance)) : 
    print(str(data_feature_names[i])+"\t:\t"+ str(importance[i]*100))

    # B. Validacion del Modelo (Matriz de Confusion): 
    #   Variable de Interes es la Fuga (Exied=1)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score,roc_auc_score
cm=confusion_matrix(y_test,y_pred) # lo observado en test VS lo predicho con X_test
Exito=accuracy_score(y_test,y_pred) #(1378+154)/1873  :0.7960725075528701
Fracaso=1-Exito # El modelo asierta en el 81% de los casos
Sensibilidad=154/(154+167) # 0.4797507788161994 solo le esto acertando al 47% de los que me interesa
Epecificidad=1378/(1378+174)# 0.8878865979381443 estamos clasificando al 88% sin el valor de interes
ROC=roc_auc_score(y_test,y_pred) # 0.6807107901176099 : Cercano a 1 es mucho mejor
    
    # C. Validar el Sobre ajuste (overfitting) : cobre entrenado y no preparado para nuevos casos
y_pred_train=classifier.predict(X_train)    
Exito_overfitting=accuracy_score(y_train,y_pred_train) # 1.0 : el modelo esta 100% entrenado para el train

#==============================================
# 3. MULTIPLES ENTRENAMIENTOS
#==============================================
    # A. Varios entrenamientos 
train_pred=[]
test_pred=[]
max_deep_list=list(range(3,30)) # numero de los niveles posibles a probar
max_deep_list

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

for deep in max_deep_list:
    arbol=DecisionTreeClassifier(criterion="entropy", max_depth=deep,random_state=0)
    arbol.fit(X_train,y_train)
    train_pred.append(arbol.score(X_train,y_train)) # con el X_train saca un y_pred_train
    test_pred.append(arbol.score(X_test,y_test))# con el X_test saca un y_pred_test
    
    
    # B. grafica de resultados
plt.plot(max_deep_list,train_pred, color='r', label="Entrenamiento (Train)")
plt.plot(max_deep_list,test_pred, color='b', label="Evaluacion (Test)")
plt.title("Grafico del ajuste del Arbol de decisión")
plt.legend()
plt.ylabel("Exito")
plt.xlabel("Nro de Niveles")
plt.show()
#==============================================
# 4. MODELO FINAL (6 niveles)
#==============================================
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", max_depth=6,random_state=0)
classifier.fit(X_train,y_train)
# Acerca del modelo (Metricas)
classifier.get_params()
    # A. Importancia de las Variables
importance=classifier.feature_importances_
#importance=np.sort(importance)
matplotlib.rcParams.update({'font.size':15})
plt.barh(data_feature_names,importance*100)

for i in range(0,len(importance)) : 
    print(str(data_feature_names[i])+"\t:\t"+ str(importance[i]*100))

    # B. Validacion del Modelo (Matriz de Confusion): 
    #   Variable de Interes es la Fuga (Exied=1)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score,roc_auc_score
cm=confusion_matrix(y_test,y_pred) # lo observado en test VS lo predicho con X_test
Exito=accuracy_score(y_test,y_pred) 
Fracaso=1-Exito # El modelo asierta en el 87% de los casos
Sesibilidad=114/(114+29) # de 47%  lo llevo a 79% de aciertos a los que me interesa
Epecificidad=1516/(1516+214)# 88% lo llevamis a 87% de aciertos sin el valor de interes
ROC=roc_auc_score(y_test,y_pred) # de 68% a 66% : Cercano a 1 es mucho mejor
Gini=2*ROC-1
    # C. Visualizando el arbol online: http://www.webgraphviz.com
from sklearn import tree
tree.export_graphviz(classifier, 
                     feature_names=data_feature_names,
                     out_file="D:/Cursos/Big Data & Analytics/Analytics-Python-2018/tree_PB.dot",
                     filled=True)

help(confusion_matrix)
    
    



