# Predicción de Cáncer de Pulmón con Machine Learning

Este proyecto desarrolla, entrena y compara 8 modelos de Machine Learning distintos para predecir la probabilidad de padecer cáncer de pulmón basándose en síntomas clínicos y características demográficas (edad y género).

## Modelos Implementados

Se utilizan los siguientes 8 algoritmos para clasificación, incluyendo 3 que se someten a un proceso de optimización de hiperparámetros (Top 3):

1.  **Regresión Logística**
2.  **Máquinas de Soporte Vectorial (SVM con Kernel RBF)**
3.  **K-Vecinos Más Cercanos (KNN)**
4.  **XGBoost**
5.  **LightGBM**
6.  _Opciones Optimizadas mediante Búsqueda Aleatoria (RandomizedSearchCV):_
    - **Random Forest (Opt)**
    - **CatBoost (Opt)**
    - **Red Neuronal Artificial (MLP-Opt)**

## Dataset (`CancerPulmonData.csv`)

El conjunto de datos debe contener, al menos, las siguientes variables:

- `GENDER`: Género del paciente.
- `AGE`: Edad.
- `LUNG_CANCER`: Variable objetivo (`YES` o `NO`).
- Otras columnas numéricas correspondientes a síntomas diversos (usualmente codificados como 1=Ausencia, 2=Presencia, las cuales el código recodifica a 0 y 1).

## Funciones Principales de `PruebaCancerDePulmon5_Optimizacion.py`

- **Preprocesamiento:**
  - Limpieza de espacios en los nombres de las columnas.
  - Conversión de variables categóricas (Género, variable objetivo) a numéricas usando `LabelEncoder` y mapeo.
  - Recodificación de las variables de síntomas y condiciones médicas.
  - Escalado de características con `StandardScaler`.
- **Entrenamiento y Optimización:**
  - División de datos usando muestreo estratificado (80% entrenamiento, 20% prueba).
  - Optimización de hiperparámetros para Random Forest, CatBoost y Redes Neuronales (MLP) para mejorar su rendimiento de línea base.
- **Evaluación Metrics:**
  - La precisión de clasificación estándar (`Accuracy`).
  - Área bajo la curva ROC (`AUC`).
- **Visualización Científica (Gráficos para reportes):**
  - _Comparativa de Rendimiento:_ Un gráfico de barras que compara Accuracy vs AUC para cada modelo.
  - _Curvas ROC Comparativas:_ Superposición de todas las curvas ROC en un único gráfico para evaluar la sensibilidad frente a la especificidad.
  - _Matrices de Confusión:_ Un mosaico que muestra la matriz de confusión (Verdaderos Positivos, Falsos Positivos, etc.) generada individualmente por cada uno de los 8 aproximaciones.
  - _Importancia de las Variables:_ Dos gráficos de barras que exhiben qué síntomas influyen más en las predicciones en los modelos de árbol más sofisticados resultantes (Random Forest vs CatBoost).

## Requisitos

El script depende de las siguientes bibliotecas que se deben instalar si no se tienen:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

## Ejecución

Para correr el script y ver los resultados analíticos por consola seguido por los gráficos generados en el entorno, simplemente ejecutar:

```bash
python PruebaCancerDePulmon5_Optimizacion.py
```

> Nota: El entrenamiento puede tardar un par de minutos, especialmente durante la búsqueda de hiperparámetros.
