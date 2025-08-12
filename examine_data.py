import pandas as pd

# Lire les fichiers CSV
print("=== Train.csv ===")
train_df = pd.read_csv('Train.csv')
print("Dimensions:", train_df.shape)
print("\nAperçu des données:")
print(train_df.head())
print("\nInformations sur les colonnes:")
print(train_df.info())
print("\nValeurs manquantes par colonne:")
print(train_df.isnull().sum())

print("\n=== Test.csv ===")
test_df = pd.read_csv('Test.csv')
print("Dimensions:", test_df.shape)
print("\nAperçu des données:")
print(test_df.head())
print("\nColonnes:", test_df.columns.tolist())
