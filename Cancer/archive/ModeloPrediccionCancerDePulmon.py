import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## Entrenamiento de Modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier # <-- AÑADE ESTA LÍNEA AQUÍ

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# --- Paso 1 y 2: Carga e Información General ---
# Asegúrate de que la ruta a tu archivo sea la correcta
df = pd.read_csv(r"C:\Users\pepej\Documents\Cancer\archive\CancerPulmonData.csv")
print("Dataset cargado. Forma:", df.shape)

# --- Paso 3: Limpieza de Nombres de Columnas ---
df.columns = df.columns.str.strip().str.replace(' ', '').str.replace('.', '')
print("\nNombres de columnas limpios y estandarizados.")

# --- Paso 4: Codificación de la Variable Objetivo ---
df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({'NO': 0, 'YES': 1})
print("\nColumna 'LUNG_CANCER' codificada a 0 y 1.")

# --- ¡NUEVO! Paso 5: Preprocesamiento Completo de Características ---

# 5.1 Codificar GENDER
df['GENDER'] = df['GENDER'].replace({'M': 0, 'F': 1})
print("Columna 'GENDER' codificada a 0 y 1.")

# 5.2 Recodificar todas las demás variables de (1=NO, 2=SÍ) a (0=NO, 1=SÍ)
cols_to_recode = df.columns.drop(['AGE', 'GENDER', 'LUNG_CANCER'])
for col in cols_to_recode:
    df[col] = df[col].replace({1: 0, 2: 1})

print(f"Se recodificaron {len(cols_to_recode)} columnas de síntomas/hábitos a 0 y 1.")

# --- AHORA TUS DATOS ESTÁN COMPLETAMENTE PREPROCESADOS ---
print("\n--- Vista Previa de los Datos Listos para el Modelo ---")
print(df.head())
print("\n--- Verificación de Tipos de Datos (Todos deben ser numéricos) ---")
print(df.info())


# --- INICIA TU ANÁLISIS EXPLORATORIO (EDA) CON DATOS LIMPIOS ---
# (Tu código original, ahora numerado a partir del Paso 6)

# --- Paso 6: Análisis de la Edad ---
print(f"\nMedia de edad: {df['AGE'].mean():.2f}")
print(f"Mediana de edad: {df['AGE'].median():.2f}")
sns.histplot(df['AGE'], kde=True, bins=20)
plt.title("Distribución de Edad de los Pacientes")
plt.show()

# --- Paso 7: Relación entre Edad y Cáncer ---
sns.boxplot(x='LUNG_CANCER', y='AGE', data=df)
plt.title("Relación entre Edad y Diagnóstico de Cáncer")
plt.xticks([0, 1], ['No Cáncer', 'Cáncer'])
plt.xlabel("Diagnóstico")
plt.ylabel("Edad")
plt.show()

# --- Paso 8: Relación entre Fumar y Cáncer ---
print("\nTabla de contingencia: Fumar vs Cáncer")
# Nota: La recodificación a 0/1 cambia cómo creamos las etiquetas del índice
smoking_ct = pd.crosstab(df['SMOKING'], df['LUNG_CANCER'])
smoking_ct.columns = ['No Cáncer', 'Cáncer']
smoking_ct.index = ['No Fumador (0)', 'Fumador (1)']
print(smoking_ct)

# --- Paso 9: Visualización de la Asociación entre Fumar y Cáncer ---
plt.figure(figsize=(10, 6))
# Usamos un lambda para cambiar las etiquetas del eje x para el gráfico
ax = sns.countplot(x='SMOKING', hue='LUNG_CANCER', data=df, palette=['#3498db', '#e74c3c'])
ax.set_xticklabels(['No Fumador', 'Sí Fumador'])
plt.title("Relación entre Hábito de Fumar y Cáncer de Pulmón", fontsize=16)
plt.xlabel("¿Es fumador?", fontsize=12)
plt.ylabel("Número de Casos", fontsize=12)
plt.legend(title='Diagnóstico', labels=['No Cáncer', 'Cáncer'])
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
plt.show()

# --- Paso 10: Análisis con Rangos de Edad ---
age_bins = [0, 50, 60, 70, 100]
age_labels = ['<50', '51-60', '61-70', '>70']
df['EDAD_RANGO'] = pd.cut(df['AGE'], bins=age_bins, labels=age_labels, right=False)

print("\nTabla de contingencia: Rango de Edad vs Enfermedad Crónica vs Cáncer")
# CHRONICDISEASE también fue recodificada a 0 y 1
ct2 = pd.crosstab([df['EDAD_RANGO'], df['CHRONICDISEASE']], df['LUNG_CANCER'])
ct2.columns = ['No Cáncer', 'Cáncer']
print(ct2)




##Entrenaminto de Modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

#Paso 1
#Normalizacion de datos
# Eliminar filas donde EDAD o PREECLAMPSIA sean nulos
df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')

#Asegurar que PREECLAMPSIA sea 0 o 1 numérico
df['SMOKING'] = pd.to_numeric(df['SMOKING'], errors='coerce')

df['LUNG_CANCER'] = pd.to_numeric(df['LUNG_CANCER'], errors='coerce')

# Selección de columnas y eliminar filas con NaN
df_model = df[['AGE', 'SMOKING', 'LUNG_CANCER']].dropna().copy()


#Paso 2
# Selección de variables
X = df_model[['AGE', 'SMOKING']]  # Variables predictoras
y = df_model['LUNG_CANCER']      # Variable objetivo


#Paso 3
# Dividir en entrenamiento y prueba 80 20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20
    random_state=42,    # Para que el resultado sea reproducible
    stratify=y          # Asegura que la proporción de cáncer/no cáncer sea igual en ambos sets
)

# Paso 4 
# Creacion del modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

#Paso 5
#Evaluar el modelo

#Predicciones
y_pred = modelo.predict(X_test)
#Probabilida de que cad clase (edad + preclamsia) = aborto
y_prob = modelo.predict_proba(X_test)[:,1]

#Paso 6
#Métricas
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\n")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
print("\n")

print("AUC:", roc_auc_score(y_test, y_prob))
print("\n")

# #paso 7
# Calculamos la curva ROC
from sklearn.metrics import roc_curve

# Predecir probabilidades para la clase positiva
y_prob = modelo.predict_proba(X_test)[:,1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# AUC
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.3f}")

# Encontrar el umbral óptimo
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
print(f"Umbral óptimo: {optimal_threshold:.3f}")

# Graficar ROC
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Umbral óptimo = {optimal_threshold:.3f}')
plt.xlabel('Falso positivo')
plt.ylabel('Verdadero positivo')
plt.title('Curva ROC')
plt.legend()
plt.show()

# --- INICIO DEL NUEVO MODELO: ÁRBOL DE DECISIÓN ---
# *****************************************************************
print("\n--- MODELO 2: ÁRBOL DE DECISIÓN ---")

# 1. Crear y entrenar el modelo de Árbol de Decisión
# Se usan los mismos datos (X_train, y_train) para una comparación justa
arbol_modelo = DecisionTreeClassifier(random_state=42)
arbol_modelo.fit(X_train, y_train)

# 2. Hacer predicciones
y_pred_arbol = arbol_modelo.predict(X_test)
y_prob_arbol = arbol_modelo.predict_proba(X_test)[:, 1]

# 3. Evaluar el modelo
print("\nMatriz de confusión (Árbol de Decisión):")
print(confusion_matrix(y_test, y_pred_arbol))
print("\n")

print("\nReporte de clasificación (Árbol de Decisión):")
print(classification_report(y_test, y_pred_arbol))
print("\n")

# Calcular y mostrar Accuracy y AUC
accuracy_arbol = accuracy_score(y_test, y_pred_arbol)
auc_arbol = roc_auc_score(y_test, y_prob_arbol)

print(f"Accuracy (Árbol de Decisión): {accuracy_arbol:.3f}")
print(f"AUC (Árbol de Decisión): {auc_arbol:.3f}")

# 4. Visualizar el árbol
plt.figure(figsize=(15, 10))
plot_tree(arbol_modelo,
          feature_names=X.columns.tolist(),
          class_names=['No Cáncer', 'Cáncer'],
          filled=True,
          rounded=True)
plt.title("Visualización del Árbol de Decisión")
plt.show()
# *****************************************************************
# --- FIN DEL NUEVO MODELO ---
# *****************************************************************

