import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def data_loading():
    path = kagglehub.dataset_download("sulianova/cardiovascular-disease-dataset")
    print("Path to dataset files:", path)

    files = os.listdir(path)
    csv_file = [f for f in files if f.endswith('.csv')][0]
    full_path = os.path.join(path, csv_file)

    df = pd.read_csv(full_path, sep=';')

    print("\n--- Aperçu des 5 premières lignes ---")
    print(df.head(25))

    print("\n--- Informations structurelles (Types et Valeurs Manquantes) ---")
    print(df.info())
    return df

def train_model(df):
    X = df.drop(['id', 'cardio'], axis=1)
    y = df['cardio']

    # Séparation Entraînement 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création et entraînement du modèle Baseline
    model_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_baseline.fit(X_train, y_train)

    # Prédiction
    y_pred = model_baseline.predict(X_test)

    # Évaluation
    print("\n--- RÉSULTATS DU MODÈLE BASELINE (DONNÉES BRUTES) ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nMatrice de Confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport de Classification :")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    df = data_loading()
    train_model(df)