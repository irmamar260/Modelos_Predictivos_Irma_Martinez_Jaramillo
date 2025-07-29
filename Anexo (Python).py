# Importación de bibliotecas
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2

# 1. Carga y limpieza de datos
def load_and_clean_data():
    # Cargar el dataset
    adult = fetch_ucirepo(id=2)
    df = pd.concat([adult.data.features, adult.data.targets], axis=1)
    
    # Verificación básica
    print("Dimensiones del dataset:", df.shape)
    print("\nVariables disponibles:\n", df.columns.tolist())
    print("\nMuestra de datos:\n", df.head())

    # Limpieza inicial
    def clean_data(df):
        # Eliminar espacios y valores desconocidos
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df.replace('?', np.nan, inplace=True)
        
        # Limpieza específica del target
        df['income'] = df['income'].str.rstrip('.').map({'<=50K': 0, '>50K': 1})
        
        # Manejo de valores faltantes
        return df.dropna()
    
    df_clean = clean_data(df)
    df_clean.to_csv('C:/Users/irmam/Downloads/adult_clean.csv', index=False)
    
    # Análisis de datos limpios
    print("\nDatos faltantes después de limpieza:\n", df_clean.isnull().sum())
    print("\nDistribución del target:\n", df_clean['income'].value_counts(normalize=True))
    
    return df_clean

# 2. Análisis exploratorio
def exploratory_analysis(df_clean):
    # Análisis descriptivo
    print("\nEstadísticas descriptivas numéricas:\n", df_clean.select_dtypes(include=['int64', 'float64']).describe())
    print("\nEstadísticas descriptivas categóricas:\n", df_clean.select_dtypes(include='object').describe())

    # Visualizaciones
    plt.figure(figsize=(15, 10))
    
    # Distribución de edad por ingreso
    plt.subplot(2, 2, 1)
    sns.boxplot(x='income', y='age', data=df_clean)
    plt.title('Distribución de Edad por Nivel de Ingreso')
    
    # Distribución de horas trabajadas
    plt.subplot(2, 2, 2)
    sns.histplot(df_clean['hours-per-week'], bins=30)
    plt.title('Distribución de Horas Trabajadas por Semana')
    
    # Relación entre educación e ingreso
    plt.subplot(2, 2, 3)
    sns.countplot(y='education', hue='income', data=df_clean, order=df_clean['education'].value_counts().index)
    plt.title('Nivel de Educación vs Ingreso')
    
    # Correlación numérica
    plt.subplot(2, 2, 4)
    numeric_df = df_clean.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    
    plt.tight_layout()
    plt.show()

# 3. Preparación de datos para modelado
def prepare_data(df_clean):
    X = df_clean.drop('income', axis=1)
    y = df_clean['income']
    
    # Codificación de variables categóricas
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Selección de características
    # Método 1: SelectKBest con chi-cuadrado
    selector_chi2 = SelectKBest(chi2, k=10)
    selector_chi2.fit(X_encoded, y)
    chi2_scores = pd.DataFrame({'Feature': X_encoded.columns, 'Chi2_Score': selector_chi2.scores_})
    chi2_scores = chi2_scores.sort_values('Chi2_Score', ascending=False)
    
    # Método 2: Importancia con Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_encoded, y)
    rf_importances = pd.DataFrame({'Feature': X_encoded.columns, 'RF_Importance': rf.feature_importances_})
    rf_importances = rf_importances.sort_values('RF_Importance', ascending=False)
    
    print("\nTop 10 características por Chi2:\n", chi2_scores.head(10))
    print("\nTop 10 características por Random Forest:\n", rf_importances.head(10))
    
    # Selección final de variables
    selected_features = list(set(chi2_scores['Feature'].head(10).tolist() + rf_importances['Feature'].head(10).tolist()))
    print("\nVariables seleccionadas:\n", selected_features)
    
    X_selected = X_encoded[selected_features]
    
    # División train-test
    return train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)

# 4. Modelado y evaluación
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Escalado de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos a evaluar
    models = {
        'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    # Evaluación de modelos
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", cm)
    
    # Visualización comparativa
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies)
    plt.title('Comparación de Accuracy entre Modelos')
    plt.ylabel('Accuracy')
    plt.ylim(0.7, 0.9)
    plt.show()

# Flujo principal de ejecución
def main():
    # 1. Carga y limpieza
    df_clean = load_and_clean_data()
    
    # 2. Análisis exploratorio
    exploratory_analysis(df_clean)
    
    # 3. Preparación de datos
    X_train, X_test, y_train, y_test = prepare_data(df_clean)
    
    # 4. Modelado y evaluación
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
    