import kagglehub
import pandas as pd
import os

from IPython.core.display_functions import display

path = kagglehub.dataset_download("sulianova/cardiovascular-disease-dataset")
print("Path to dataset files:", path)

files = os.listdir(path)
csv_file = [f for f in files if f.endswith('.csv')][0]
full_path = os.path.join(path, csv_file)

df = pd.read_csv(full_path,sep=';',decimal=',',quotechar='"', quoting=3, encoding='utf-8')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
display(df.head())


print("\n--- Informations structurelles (Types et Valeurs Manquantes) ---")
print(df.info())