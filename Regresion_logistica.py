#Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, r2_score, f1_score

# Crear una función para aplicar la lógica
def categorizar_precio(valor, media):
    if valor > media:
        return 'precio mayor'
    else:
        return 'precio menor'

#Archivos
credicel = pd.read_excel("Copia de df_limpio.xlsx")

#Limpieza
credicel["riesgo"] = credicel["riesgo"].fillna(credicel["riesgo"].median())
credicel["monto_financiado"] = credicel["monto_financiado"].fillna(credicel["monto_financiado"].median())
credicel["costo_total"] = credicel["costo_total"].fillna(credicel["costo_total"].median())
credicel["precio"] = credicel["precio"].fillna(credicel["precio"].median())
credicel["edad_cliente"] = credicel["edad_cliente"].fillna(credicel["edad_cliente"].median())
credicel["monto_accesorios"] = credicel["monto_accesorios"].fillna(credicel["monto_accesorios"].median())

#Cambiamos a categóricas
    #Riesgo
credicel["riesgo"] = pd.to_numeric(credicel["riesgo"], errors='coerce')
media_riesgo = credicel["riesgo"].mean()
valor_limite = media_riesgo
credicel["riesgo"] = credicel["riesgo"].apply(lambda x: "mayor" if x > valor_limite else "menor")

    #monto_financiado
credicel["monto_financiado"] = pd.to_numeric(credicel["monto_financiado"], errors='coerce')
media_monto_financiado = credicel["monto_financiado"].mean()
valor_limite = media_monto_financiado
credicel["monto_financiado"] = credicel["monto_financiado"].apply(lambda x: "mayor" if x > valor_limite else "menor")

    #costo_total
credicel["costo_total"] = pd.to_numeric(credicel["costo_total"], errors='coerce')
media_costo_total = credicel["costo_total"].mean()
valor_limite = media_costo_total
credicel["costo_total"] = credicel["costo_total"].apply(lambda x: "mayor" if x > valor_limite else "menor")

#Declaramos variables dependientes
y_riesgo = credicel["riesgo"]
y_monto_financiado = credicel["monto_financiado"]
y_costo_total = credicel["costo_total"]

#Definición de variables X
x_riesgo = credicel[["precio", "edad_cliente"]]
x_monto_financiado = credicel[["precio", "monto_accesorios"]]
x_costo_total = credicel[["precio", "monto_accesorios"]]

#___________________________________________________________________________________________________________________________________________________

#Creación de Modelos
#RIESGO
print("RIESGO--------------------------------------------------------------------------------------------------------")
X_train_riesgo, X_test_riesgo, y_train_riesgo, y_test_riesgo = train_test_split(x_riesgo, y_riesgo, test_size = 0.45, random_state = None)
escalar = StandardScaler()
X_train_riesgo = escalar.fit_transform(X_train_riesgo)
X_test_riesgo = escalar.transform(X_test_riesgo)
algoritmo = LogisticRegression()
algoritmo.fit(X_train_riesgo, y_train_riesgo)
print(algoritmo.fit)
y_pred_riesgo = algoritmo.predict(X_test_riesgo)
print(y_pred_riesgo)
matriz_riesgo = confusion_matrix(y_test_riesgo, y_pred_riesgo)
print("Matriz de Confusión")
print(matriz_riesgo)
precision_riesgo = precision_score(y_test_riesgo, y_pred_riesgo, average = "binary", pos_label = "mayor")
print("Precisión: ", precision_riesgo)
exactitud_riesgo = accuracy_score(y_test_riesgo, y_pred_riesgo)
print("Exactitud: ", exactitud_riesgo)
sensibilidad_riesgo = recall_score(y_test_riesgo, y_pred_riesgo, average = "binary", pos_label = "mayor") 
print("Sensibilidad:", sensibilidad_riesgo)
puntajef1_riesgo = f1_score(y_test_riesgo, y_pred_riesgo, average = "binary", pos_label = "mayor")
print("Puntaje F1: ", puntajef1_riesgo)

#MONTO FINANCIADO
print("MONTO FINANCIADO---------------------------------------------------------------------------------------------------------------------")
X_train_monto_financiado, X_test_monto_financiado, y_train_monto_financiado, y_test_monto_financiado = train_test_split(x_monto_financiado, y_monto_financiado, test_size = 0.45, random_state = None)
escalar = StandardScaler()
X_train_monto_financiado = escalar.fit_transform(X_train_monto_financiado)
X_test_monto_financiado = escalar.transform(X_test_monto_financiado)
algoritmo = LogisticRegression()
algoritmo.fit(X_train_monto_financiado, y_train_monto_financiado)
print(algoritmo.fit)
y_pred_monto_financiado = algoritmo.predict(X_test_monto_financiado)
print(y_pred_monto_financiado)
matriz_monto_financiado = confusion_matrix(y_test_monto_financiado, y_pred_monto_financiado)
print("Matriz de Confusión")
print(matriz_monto_financiado)
precision_monto_financiado = precision_score(y_test_monto_financiado, y_pred_monto_financiado, average = "binary", pos_label = "mayor")
print("Precisión: ", precision_monto_financiado)
exactitud_monto_financiado = accuracy_score(y_test_monto_financiado, y_pred_monto_financiado)
print("Exactitud: ", exactitud_monto_financiado)
sensibilidad_monto_financiado = recall_score(y_test_monto_financiado, y_pred_monto_financiado, average = "binary", pos_label = "mayor") 
print("Sensibilidad:", sensibilidad_monto_financiado)
puntajef1_monto_financiado = f1_score(y_test_monto_financiado, y_pred_monto_financiado, average = "binary", pos_label = "mayor")
print("Puntaje F1: ", puntajef1_monto_financiado)

#COSTO TOTAL
print("COSTO TOTAL-----------------------------------------------------------------------------------------------------------")
X_train_costo_total, X_test_costo_total, y_train_costo_total, y_test_costo_total = train_test_split(x_costo_total, y_costo_total, test_size = 0.45, random_state = None)
escalar = StandardScaler()
X_train_costo_total = escalar.fit_transform(X_train_costo_total)
X_test_costo_total = escalar.transform(X_test_costo_total)
algoritmo = LogisticRegression()
algoritmo.fit(X_train_costo_total, y_train_costo_total)
print(algoritmo.fit)
y_pred_costo_total = algoritmo.predict(X_test_costo_total)
print(y_pred_costo_total)
matriz_costo_total = confusion_matrix(y_test_costo_total, y_pred_costo_total)
print("Matriz de Confusión")
print(matriz_costo_total)
precision_costo_total = precision_score(y_test_costo_total, y_pred_costo_total, average = "binary", pos_label = "mayor")
print("Precisión: ", precision_costo_total)
exactitud_costo_total = accuracy_score(y_test_costo_total, y_pred_costo_total)
print("Exactitud: ", exactitud_costo_total)
sensibilidad_costo_total = recall_score(y_test_costo_total, y_pred_costo_total, average = "binary", pos_label = "mayor") 
print("Sensibilidad:", sensibilidad_costo_total)
puntajef1_costo_total = f1_score(y_test_costo_total, y_pred_costo_total, average = "binary", pos_label = "mayor")
print("Puntaje F1: ", puntajef1_costo_total)