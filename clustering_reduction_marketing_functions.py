"""
Module contenant les fonctions pour le projet de clustering et réduction de dimensions marketing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configuration des graphiques
sns.set_theme(style='darkgrid', palette='viridis')


def load_data(train_path='Train.csv', test_path='Test.csv'):
    """
    Charge les données d'entraînement et de test.
    
    Args:
        train_path (str): Chemin vers le fichier d'entraînement
        test_path (str): Chemin vers le fichier de test
        
    Returns:
        tuple: (train_df, test_df) - DataFrames d'entraînement et de test
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None, None


def explore_data(df, name=""):
    """
    Affiche des informations de base sur le DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
        name (str): Nom du DataFrame pour l'affichage
    """
    if name:
        print(f"\n=== Exploration de {name} ===")
    
    print(f"\nDimensions: {df.shape}")
    print("\nAperçu des données:")
    display(df.head())
    
    print("\nRésumé statistique:")
    display(df.describe(include='all'))
    
    print("\nTypes de données:")
    print(df.dtypes)
    
    print("\nValeurs manquantes par colonne:")
    print(df.isnull().sum())
    
    # Analyse des variables catégorielles
    cat_cols = df.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        print("\nValeurs uniques par colonne catégorielle:")
        for col in cat_cols:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).head())


def preprocess_data(df, is_train=True, label_encoders=None):
    """
    Prétraite les données pour l'analyse.
    
    Args:
        df (pd.DataFrame): Données à prétraiter
        is_train (bool): Si True, traite les données d'entraînement
        label_encoders (dict): Dictionnaire des encodeurs pour les variables catégorielles
        
    Returns:
        tuple: (pd.DataFrame, dict) - Données prétraitées et encodeurs
    """

    df_processed = df.copy()

    # Supprimer la colonne ID
    if 'ID' in df_processed.columns:
        df_processed.drop('ID', axis=1, inplace=True)

    # Initialiser le dictionnaire d'encodeurs
    if is_train or label_encoders is None:
        label_encoders = {}

    # Traitement des valeurs manquantes pour les variables numériques
    num_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    if not num_cols.empty:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[num_cols] = num_imputer.fit_transform(df_processed[num_cols])

    # Traitement des valeurs manquantes pour les variables catégorielles
    cat_cols = df_processed.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[cat_cols] = cat_imputer.fit_transform(df_processed[cat_cols])

        # Encodage des variables catégorielles
        for col in cat_cols:
            df_processed[col] = df_processed[col].astype(str)
            if is_train:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                label_encoders[col] = le
            else:
                if col in label_encoders:
                    le = label_encoders[col]
                    known_classes = set(le.classes_)
                    # Ajouter les nouvelles classes au label encoder
                    new_classes = set(df_processed[col].unique()) - known_classes
                    if new_classes:
                        le.classes_ = np.append(le.classes_, list(new_classes))
                    df_processed[col] = le.transform(df_processed[col])
                else:
                    # Si le label encoder n'existe pas, encoder comme inconnu
                    df_processed[col] = -1

    # Standardisation des données numériques
    if not num_cols.empty:
        scaler = StandardScaler()
        df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

    return df_processed, label_encoders
    

def determine_optimal_clusters(data, max_clusters=10):
    """
    Détermine le nombre optimal de clusters avec la méthode du coude.
    
    Args:
        data (pd.DataFrame): Données à analyser
        max_clusters (int): Nombre maximum de clusters à tester
        
    Returns:
        tuple: (inertias, silhouette_scores) - Inerties et scores de silhouette pour chaque k
    """
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        
        inertias.append(kmeans.inertia_)
        
        # Le score de silhouette nécessite au moins 2 clusters et moins de clusters que d'échantillons
        if 1 < k < len(data):
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append((k, score))
    
    # Tracer la courbe du coude
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertias, 'bo-')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude')
    
    # Tracer le score de silhouette
    if silhouette_scores:
        k_values, scores = zip(*silhouette_scores)
        plt.subplot(1, 2, 2)
        plt.plot(k_values, scores, 'ro-')
        plt.xlabel('Nombre de clusters (k)')
        plt.ylabel('Score de silhouette')
        plt.title('Score de silhouette')
    
    plt.tight_layout()
    plt.show()
    
    return inertias, silhouette_scores


def apply_clustering(data, n_clusters=4, method='kmeans'):
    """
    Applique un algorithme de clustering aux données.
    
    Args:
        data (pd.DataFrame): Données à regrouper
        n_clusters (int): Nombre de clusters à former
        method (str): Méthode de clustering ('kmeans', 'hierarchical', 'dbscan')
        
    Returns:
        tuple: (labels, model) - Étiquettes des clusters et modèle ajusté
    """
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError("Méthode de clustering non reconnue. Choisissez parmi 'kmeans', 'hierarchical', 'dbscan'")
    
    labels = model.fit_predict(data)
    return labels, model


def reduce_dimensionality(data, method='pca', n_components=2):
    """
    Réduit la dimensionnalité des données.
    
    Args:
        data (pd.DataFrame): Données à réduire
        method (str): Méthode de réduction ('pca' ou 'tsne')
        n_components (int): Nombre de dimensions cibles
        
    Returns:
        np.ndarray: Données réduites
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30, max_iter=300)
    else:
        raise ValueError("Méthode de réduction non reconnue. Choisissez 'pca' ou 'tsne'")
    
    return reducer.fit_transform(data)


def plot_clusters(data_2d, labels, title='Visualisation des clusters'):
    """
    Affiche les clusters dans un espace 2D.
    
    Args:
        data_2d (np.ndarray): Données en 2D
        labels (np.ndarray): Étiquettes des clusters
        title (str): Titre du graphique
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title(title)
    plt.xlabel('Composante 1')
    plt.ylabel('Composante 2')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def evaluate_clustering(data, labels):
    """
    Évalue la qualité du clustering.
    
    Args:
        data (pd.DataFrame): Données d'origine
        labels (np.ndarray): Étiquettes des clusters
        
    Returns:
        dict: Métriques d'évaluation
    """
    if len(set(labels)) < 2:
        return {
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None,
            'n_clusters': len(set(labels))
        }
    
    metrics = {
        'silhouette': silhouette_score(data, labels),
        'calinski_harabasz': calinski_harabasz_score(data, labels),
        'davies_bouldin': davies_bouldin_score(data, labels),
        'n_clusters': len(set(labels))
    }
    
    print("\n=== Métriques d'évaluation ===")
    print(f"Nombre de clusters: {metrics['n_clusters']}")
    print(f"Score de silhouette: {metrics['silhouette']:.3f} (plus proche de 1 est meilleur)")
    print(f"Indice de Calinski-Harabasz: {metrics['calinski_harabasz']:.2f} (plus élevé est meilleur)")
    print(f"Indice de Davies-Bouldin: {metrics['davies_bouldin']:.3f} (plus proche de 0 est meilleur)")
    
    return metrics
