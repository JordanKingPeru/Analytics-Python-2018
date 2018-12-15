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
ds=pd.read_csv("D:/BDA_VIII/Clase_07/01.TratamientoDatos" +
                "/SegmentoPaisesBajos_PB/ds1_PB.csv")
ds=ds.drop(["Unnamed: 0"], axis=1)
ds.info()

ds=ds.drop(["Geography"], axis=1) # Eliminamos la variable segmentadora

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

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# Acerca del modelo (Metricas)
classifier.get_params()
    # A. Importancia de las Variables
feature_importance=classifier.coef_[0]
feature_importance=100*(feature_importance/feature_importance.max())
sorted_idx=np.argsort(feature_importance)
feature_name=np.asarray(data_feature_names)
feature_name
feature_name[sorted_idx]
feature_importance[sorted_idx]

#******** por revisar
plt.barh(feature_name[sorted_idx],feature_importance[sorted_idx])
plt.xlabel("Realtive Feature Importance")
plt.show()

    # B. Validacion del Modelo (Matriz de Confusion): 
    #   Variable de Interes es la Fuga (Exied=1)
prob_pred_test=classifier.predict_proba(X_test)
prob_pred_test=pd.DataFrame(prob_pred_test)
plt.hist(prob_pred_test[1])
y_pred=np.where(prob_pred_test[1]>=0.25,1,0) #y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score,roc_auc_score
cm=confusion_matrix(y_test,y_pred) # lo observado en test VS lo predicho con X_test
Exito=accuracy_score(y_test,y_pred) #0.77
Fracaso=1-Exito # El modelo asierta en el 77% de los casos
Sesibilidad=77/(77+179) # 0.30078125 solo le estoy acertando al 30% de los que me interesa
Epecificidad=1366/(1366+251)# 0.8447742733457019 estamos clasificando al 84% sin el valor de interes
ROC=roc_auc_score(y_test,y_pred) # 0.5594492461914912 : Cercano a 1 es mucho mejor
    
    # C. Validar el Sobre ajuste (overfitting) : sobre entrenado y no preparado para nuevos casos
#y_pred_train=classifier.predict(X_train) 
prob_pred_train=classifier.predict_proba(X_train)
prob_pred_train=pd.DataFrame(prob_pred_train)
y_pred_train=np.where(prob_pred_train[1]>=0.25,1,0)
   
Exito_overfitting=accuracy_score(y_train,y_pred_train) # 0.79 : el modelo no esta sobre entrenado

#==============================================
# 3. MULTIPLES ENTRENAMIENTOS
#==============================================
    # A. Varios entrenamientos 
train_pred=[]
test_pred=[]
C_param_range=[0.001,0.01,0.1,1,10,100,1000] # Niveles de complejidad para convergencia al minimo error

from sklearn.metrics import accuracy_score
c=0
for c in C_param_range:
    modelo=LogisticRegression(C=c,random_state=0)
    modelo.fit(X_train,y_train)
    
    prob_pred_train=modelo.predict_proba(X_train)
    prob_pred_train=pd.DataFrame(prob_pred_train)
    y_pred_train=np.where(prob_pred_train[1]>=0.25,1,0)
    train_pred.append(accuracy_score(y_pred_train,y_train)) # con el X_train saca un y_pred_train
    
    prob_pred_test=modelo.predict_proba(X_test)
    prob_pred_test=pd.DataFrame(prob_pred_test)
    y_pred_test=np.where(prob_pred_test[1]>=0.25,1,0)
    test_pred.append(accuracy_score(y_pred_test,y_test))# con el X_test saca un y_pred_test
    
    
    # B. grafica de resultados
plt.plot(C_param_range,train_pred, color='r', label="Entrenamiento (Train)")
plt.plot(C_param_range,test_pred, color='b', label="Evaluacion (Test)")

x=np.asanyarray(train_pred)-np.asanyarray(test_pred)
plt.plot(C_param_range,x, color='b', label="Evaluacion (Test)")
plt.title("Grafico del ajuste del Arbol de decisión")
plt.legend()
plt.ylabel("Exito")
plt.xlabel("Nro de Niveles")
plt.show()
#==============================================
# 4. MODELO FINAL (quira la vartiable menos importante...... )
#==============================================
## TBD    
    
    
    


