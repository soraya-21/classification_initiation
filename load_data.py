import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("sulianova/cardiovascular-disease-dataset")
print("Path to dataset files:", path)

files = os.listdir(path)
csv_file = [f for f in files if f.endswith('.csv')][0]
full_path = os.path.join(path, csv_file)

df = pd.read_csv(full_path)

print("\n--- Aperçu des 5 premières lignes ---")
print(df.head())

print("\n--- Informations structurelles (Types et Valeurs Manquantes) ---")
print(df.info())