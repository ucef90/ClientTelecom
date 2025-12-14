# test_pipeline_phase1.py
import os
import pandas as pd

# S'assurer que Python peut trouver le package src
import sys
sys.path.append(os.path.abspath("src"))

from data.load_data import load_data
from data.preprocess import preprocess_data
from features.build_features import build_features

# ==========================================================
# CONFIGURATION
# ==========================================================
# Chemin vers le fichier CSV brut (à adapter selon ta machine)
DATA_PATH = "/Users/riadanas/Desktop/Telco Customer Churn MLE/data/raw/Telco-Customer-Churn.csv"  # ajuste selon ton chemin local

# Nom de la colonne cible
TARGET_COL = "Churn"


def main():
    print("=== Test Phase 1 : Chargement → Prétraitement → Feature Engineering ===")

    # ==========================================================
    # 1) Chargement des données
    # ==========================================================
    print("\n[1] Chargement des données...")
    df = load_data(DATA_PATH)
    print(f"Données chargées. Shape : {df.shape}")
    print(df.head(3))

    # ==========================================================
    # 2) Prétraitement
    # ==========================================================
    # Nettoyage des données (ex : suppression ID, correction types, NA, etc.)
    print("\n[2] Prétraitement des données...")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"Données après prétraitement. Shape : {df_clean.shape}")
    print(df_clean.head(3))

    # ==========================================================
    # 3) Feature Engineering
    # ==========================================================
    # Transformation en variables exploitables par le modèle :
    # - encodage binaire (0/1) pour les variables à 2 modalités
    # - one-hot encoding pour les variables multi-catégories
    print("\n[3] Construction des features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Données après feature engineering. Shape : {df_features.shape}")
    print(df_features.head(3))

    print("\n✅ Phase 1 du pipeline exécutée avec succès !")


if __name__ == "__main__":
    main()


# =====================================================================
# EXPLICATION GLOBALE – TEST PIPELINE PHASE 1 (MLOps / DEV)
# =====================================================================
#
# Objectif de ce script :
# - Valider rapidement que la première partie du pipeline fonctionne
#   correctement, étape par étape, avant d’enchaîner sur le training.
#
# Ce script vérifie :
# 1) load_data :
#    - le CSV est accessible
#    - les données sont chargées correctement
#
# 2) preprocess_data :
#    - nettoyage de base
#    - conversion de types (ex: TotalCharges)
#    - gestion simple des valeurs manquantes
#    - encodage de la cible si nécessaire
#
# 3) build_features :
#    - transformation des catégories en variables numériques
#    - génération des colonnes finales attendues par le modèle
#
# Pourquoi c'est important :
# - Détecte très tôt les erreurs de chemin, de schéma, ou de types
# - Évite de perdre du temps en lançant un entraînement qui échoue tard
# - Sert de "smoke test" avant intégration CI/CD
#
# Bonnes pratiques (pour industrialisation) :
# - Remplacer DATA_PATH absolu par :
#   - un chemin relatif (ex: data/raw/Telco-Customer-Churn.csv)
#   - ou une variable d’environnement (DATA_PATH)
#
# Évolution possible :
# - convertir ce script en test automatisé (pytest)
# - ajouter des assertions sur la forme (shape), colonnes attendues,
#   absence de NaN, etc.
