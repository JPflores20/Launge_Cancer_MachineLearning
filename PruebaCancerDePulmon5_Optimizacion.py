import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

nombre_archivo = 'CancerPulmonData.csv'
dataframe = pd.read_csv(nombre_archivo)

dataframe.columns = dataframe.columns.str.strip()

label_encoder_gender = LabelEncoder()
dataframe['GENDER'] = label_encoder_gender.fit_transform(dataframe['GENDER'])

dataframe['LUNG_CANCER'] = dataframe['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

columnas_sintomas = [columna for columna in dataframe.columns if columna not in ['GENDER', 'AGE', 'LUNG_CANCER']]
for columna in columnas_sintomas:
    dataframe[columna] = dataframe[columna].replace({1: 0, 2: 1})

caracteristicas_x = dataframe.drop('LUNG_CANCER', axis=1)
etiquetas_y = dataframe['LUNG_CANCER']

caracteristicas_entrenamiento, caracteristicas_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
    caracteristicas_x, etiquetas_y, test_size=0.2, random_state=42, stratify=etiquetas_y
)

escalador = StandardScaler()
caracteristicas_entrenamiento_escaladas = pd.DataFrame(escalador.fit_transform(caracteristicas_entrenamiento), columns=caracteristicas_x.columns)
caracteristicas_prueba_escaladas = pd.DataFrame(escalador.transform(caracteristicas_prueba), columns=caracteristicas_x.columns)

modelo_regresion_logistica = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
modelo_maquina_soporte_vectorial = SVC(probability=True, random_state=42, class_weight='balanced')
modelo_k_vecinos = KNeighborsClassifier(n_neighbors=5)
modelo_xgboost = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
modelo_lightgbm = lgb.LGBMClassifier(random_state=42, verbose=-1)

print("--- INICIANDO BÚSQUEDA DE HIPERPARÁMETROS PARA LOS TOP 3 MODELOS ---")
print("Esto puede tardar un par de minutos...\n")

parametros_random_forest = {'n_estimators': [100, 200, 300, 500], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
parametros_catboost = {'iterations': [100, 200, 300], 'depth': [4, 6, 8], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
parametros_red_neuronal = {'hidden_layer_sizes': [(64, 32), (100,), (128, 64), (50, 50)], 'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.01]}

from sklearn.base import BaseEstimator, ClassifierMixin

class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=100, depth=6, learning_rate=0.03, verbose=0, random_state=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.random_state = random_state
        self.model_ = None

    def fit(self, caracteristicas, etiquetas, **parametros_ajuste):
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            random_state=self.random_state
        )
        self.model_.fit(caracteristicas, etiquetas, **parametros_ajuste)
        return self

    def predict(self, caracteristicas):
        return self.model_.predict(caracteristicas)

    def predict_proba(self, caracteristicas):
        return self.model_.predict_proba(caracteristicas)
        
    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

busqueda_random_forest = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), parametros_random_forest, n_iter=10, cv=3, scoring='accuracy', random_state=42)
busqueda_catboost = RandomizedSearchCV(CatBoostWrapper(), parametros_catboost, n_iter=10, cv=3, scoring='accuracy', random_state=42)
busqueda_red_neuronal = RandomizedSearchCV(MLPClassifier(max_iter=1000, random_state=42), parametros_red_neuronal, n_iter=10, cv=3, scoring='accuracy', random_state=42)

print("Entrenando Random Forest...")
busqueda_random_forest.fit(caracteristicas_entrenamiento_escaladas, etiquetas_entrenamiento)
print("Entrenando CatBoost...")
busqueda_catboost.fit(caracteristicas_entrenamiento_escaladas, etiquetas_entrenamiento)
print("Entrenando Red Neuronal (MLP)...\n")
busqueda_red_neuronal.fit(caracteristicas_entrenamiento_escaladas, etiquetas_entrenamiento)

diccionario_modelos = {
    "Regresión Logística": modelo_regresion_logistica,
    "Random Forest (Opt)": busqueda_random_forest.best_estimator_,
    "SVM (Kernel RBF)": modelo_maquina_soporte_vectorial,
    "K-Vecinos (KNN)": modelo_k_vecinos,
    "XGBoost": modelo_xgboost,
    "LightGBM": modelo_lightgbm,
    "CatBoost (Opt)": busqueda_catboost.best_estimator_,
    "Red Neuronal (MLP-Opt)": busqueda_red_neuronal.best_estimator_
}

resultados_modelos = []
print(f"--- ENTRENANDO Y EVALUANDO {len(diccionario_modelos)} MODELOS ---\n")

for nombre_modelo, modelo_actual in diccionario_modelos.items():
    try:
        if "Opt" not in nombre_modelo:
            modelo_actual.fit(caracteristicas_entrenamiento_escaladas, etiquetas_entrenamiento)
            
        predicciones_etiquetas = modelo_actual.predict(caracteristicas_prueba_escaladas)
        
        if hasattr(modelo_actual, "predict_proba"):
            probabilidades_prediccion = modelo_actual.predict_proba(caracteristicas_prueba_escaladas)[:, 1]
            area_bajo_curva = roc_auc_score(etiquetas_prueba, probabilidades_prediccion)
        else:
            area_bajo_curva = 0.0
            
        precision_modelo = accuracy_score(etiquetas_prueba, predicciones_etiquetas)
        
        resultados_modelos.append({'Modelo': nombre_modelo, 'Accuracy': precision_modelo, 'AUC': area_bajo_curva})
        print(f"✅ {nombre_modelo}: Accuracy = {precision_modelo:.4f} | AUC = {area_bajo_curva:.4f}")
    except Exception as error_entrenamiento:
        print(f"❌ Error con {nombre_modelo}: {error_entrenamiento}")

dataframe_resultados = pd.DataFrame(resultados_modelos).sort_values(by='Accuracy', ascending=False)
print("\n--- RESULTADOS DE LOS MODELOS ---")
print(dataframe_resultados.to_string(index=False))

sns.set_theme(style="whitegrid")

plt.figure(figsize=(14, 6))
dataframe_derretido = dataframe_resultados.melt(id_vars="Modelo", var_name="Métrica", value_name="Score")
sns.barplot(x="Modelo", y="Score", hue="Métrica", data=dataframe_derretido, palette="viridis")
plt.title("Comparativa de Rendimiento de Todos los Modelos (Accuracy vs AUC)", fontsize=14, fontweight='bold')
plt.ylim(0.7, 1.05)
plt.xticks(rotation=20, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
for nombre_modelo, modelo_actual in diccionario_modelos.items():
    if hasattr(modelo_actual, "predict_proba"):
        probabilidades_prediccion = modelo_actual.predict_proba(caracteristicas_prueba_escaladas)[:, 1]
        tasa_falsos_positivos, tasa_verdaderos_positivos, umbrales = roc_curve(etiquetas_prueba, probabilidades_prediccion)
        area_bajo_curva = roc_auc_score(etiquetas_prueba, probabilidades_prediccion)
        plt.plot(tasa_falsos_positivos, tasa_verdaderos_positivos, label=f'{nombre_modelo} (AUC = {area_bajo_curva:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Azar')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC Comparativas (Todos los Modelos)', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

numero_total_modelos = len(diccionario_modelos)
columnas_grafica = 4
filas_grafica = math.ceil(numero_total_modelos / columnas_grafica)

figura, ejes_graficos = plt.subplots(filas_grafica, columnas_grafica, figsize=(18, 4 * filas_grafica))
ejes_graficos = ejes_graficos.flatten()

for indice_modelo, (nombre_modelo, modelo_actual) in enumerate(diccionario_modelos.items()):
    predicciones_etiquetas = modelo_actual.predict(caracteristicas_prueba_escaladas)
    matriz_confusion = confusion_matrix(etiquetas_prueba, predicciones_etiquetas)
    
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ejes_graficos[indice_modelo],
                xticklabels=['Sano', 'Cáncer'], yticklabels=['Sano', 'Cáncer'])
    ejes_graficos[indice_modelo].set_title(nombre_modelo, fontweight='bold', fontsize=11)
    ejes_graficos[indice_modelo].set_xlabel('Predicción del Modelo')
    ejes_graficos[indice_modelo].set_ylabel('Realidad')

for indice_vacio in range(indice_modelo + 1, len(ejes_graficos)):
    ejes_graficos[indice_vacio].axis('off')

plt.suptitle("Matrices de Confusión de Todos los Modelos", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\n--- CALCULANDO IMPORTANCIA DE LAS VARIABLES ---")

modelo_random_forest_optimizado = diccionario_modelos["Random Forest (Opt)"]
modelo_catboost_optimizado = diccionario_modelos["CatBoost (Opt)"]

importancias_random_forest = modelo_random_forest_optimizado.feature_importances_
importancias_catboost = modelo_catboost_optimizado.feature_importances_

dataframe_importancias = pd.DataFrame({
    'Característica': caracteristicas_x.columns,
    'RF_Importancia': importancias_random_forest,
    'CatBoost_Importancia': importancias_catboost
})

plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
dataframe_importancias_random_forest = dataframe_importancias.sort_values(by='RF_Importancia', ascending=False)
sns.barplot(x='RF_Importancia', y='Característica', data=dataframe_importancias_random_forest, palette='viridis', hue='Característica', legend=False)
plt.title('Importancia de Factores (Random Forest Optimizado)', fontsize=14, fontweight='bold')
plt.xlabel('Nivel de Importancia')
plt.ylabel('Síntoma / Condición')

plt.subplot(1, 2, 2)
dataframe_importancias_catboost = dataframe_importancias.sort_values(by='CatBoost_Importancia', ascending=False)
sns.barplot(x='CatBoost_Importancia', y='Característica', data=dataframe_importancias_catboost, palette='plasma', hue='Característica', legend=False)
plt.title('Importancia de Factores (CatBoost Optimizado)', fontsize=14, fontweight='bold')
plt.xlabel('Nivel de Importancia')
plt.ylabel('')

plt.tight_layout()
plt.show()
