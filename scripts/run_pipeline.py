#!/usr/bin/env python3
"""
Exécute séquentiellement : chargement → validation → prétraitement → feature engineering
"""

import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# === Correction du chemin d'import pour les modules locaux ===
# IMPORTANT : permet d'importer correctement les modules depuis le dossier src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Modules locaux - composants clés du pipeline
from src.data.load_data import load_data                    # Chargement des données avec gestion d'erreurs
from src.data.preprocess import preprocess_data            # Nettoyage de base
from src.features.build_features import build_features     # Feature engineering (CRITIQUE pour la performance)
from src.utils.validate_data import validate_telco_data    # Validation qualité des données


def main(args):
    """
    Fonction principale d'entraînement qui orchestre le workflow ML complet.
    """

    # ==========================================================
    # CONFIGURATION MLFLOW – SUIVI D’EXPÉRIENCES (EXPERIMENT TRACKING)
    # ==========================================================
    # MLflow sert ici à tracer chaque entraînement sous forme de "run".
    # Un run MLflow enregistre :
    # - les paramètres (hyperparamètres, seuil, test_size, etc.)
    # - les métriques (precision, recall, roc_auc, temps, etc.)
    # - les artefacts (modèle, fichiers JSON, pkl, etc.)
    #
    # Cela permet :
    # - la reproductibilité
    # - la comparaison de runs
    # - l’auditabilité (important en MLOps)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"  # Tracking local (fichiers), pas serveur
    mlflow.set_tracking_uri(mlruns_path)

    # Un "experiment" est un conteneur logique de runs (ex: "Telco Churn")
    # Si l'experiment n'existe pas, MLflow le crée automatiquement.
    mlflow.set_experiment(args.experiment)

    # Démarrage d’un run MLflow : tout ce qui est loggé dans ce bloc
    # sera rattaché à ce run (mêmes métriques, mêmes artefacts, etc.)
    with mlflow.start_run():

        # ==========================================================
        # JOURNALISATION DES PARAMÈTRES (MLflow)
        # ==========================================================
        # On log les paramètres clés pour reproduire l'expérience plus tard
        mlflow.log_param("model", "xgboost")            # Type de modèle
        mlflow.log_param("threshold", args.threshold)  # Seuil de classification utilisé
        mlflow.log_param("test_size", args.test_size)  # Ratio train/test

        # ==========================================================
        # ÉTAPE 1 : CHARGEMENT + VALIDATION QUALITÉ
        # ==========================================================
        print("🔄 Chargement des données...")
        df = load_data(args.input)
        print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

        # Validation qualité : on bloque le training si les données ne sont pas conformes
        print("🔍 Validation de la qualité des données (Great Expectations)...")
        is_valid, failed = validate_telco_data(df)

        # On log une métrique binaire : 1 si la qualité passe, 0 sinon
        # Cela permet de suivre dans le temps la stabilité de la qualité des données.
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            # En cas d'échec : on log la liste des contrôles échoués en artefact
            # (très utile pour debug / audit)
            import json
            mlflow.log_text(
                json.dumps(failed, indent=2),
                artifact_file="failed_expectations.json"
            )
            raise ValueError(f"❌ Contrôle qualité KO. Problèmes : {failed}")
        else:
            print("✅ Validation OK. Résultat loggé dans MLflow.")

        # ==========================================================
        # ÉTAPE 2 : PRÉTRAITEMENT
        # ==========================================================
        print("🔧 Prétraitement des données...")
        df = preprocess_data(df)

        # Sauvegarde du dataset prétraité pour reproductibilité / debug
        processed_path = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"✅ Dataset prétraité sauvegardé : {processed_path} | Shape : {df.shape}")

        # ==========================================================
        # ÉTAPE 3 : FEATURE ENGINEERING (CRITIQUE)
        # ==========================================================
        print("🛠️  Construction des features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Colonne cible '{target}' introuvable dans les données")

        # Encodage binaire + one-hot encoding
        df_enc = build_features(df, target_col=target)

        # Sécurité : conversion des booléens en int pour compatibilité XGBoost
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"✅ Feature engineering terminé : {df_enc.shape[1]} features")

        # ==========================================================
        # SAUVEGARDE DES MÉTADONNÉES DE FEATURES (COHÉRENCE SERVING)
        # ==========================================================
        # Objectif : garantir que l’inférence (API) utilisera EXACTEMENT
        # les mêmes colonnes et dans le même ordre que pendant le training.
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = list(df_enc.drop(columns=[target]).columns)

        # Sauvegarde locale (utile pour dev / debug)
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        # Log MLflow (artefact) : récupérable depuis l'UI MLflow
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # Artefact pkl : sert de “contrat” entre training et serving
        preprocessing_artifact = {
            "feature_columns": feature_cols,
            "target": target
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))

        # On log également ce fichier dans MLflow pour pouvoir le récupérer en prod
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"✅ Sauvegarde de {len(feature_cols)} colonnes de features pour la cohérence du serving")

        # ==========================================================
        # ÉTAPE 4 : SPLIT TRAIN / TEST
        # ==========================================================
        print("📊 Découpage des données...")
        X = df_enc.drop(columns=[target])
        y = df_enc[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            stratify=y,
            random_state=42
        )
        print(f"✅ Train : {X_train.shape[0]} échantillons | Test : {X_test.shape[0]} échantillons")

        # ==========================================================
        # GESTION DU DÉSÉQUILIBRE DE CLASSES
        # ==========================================================
        # scale_pos_weight ajuste l'importance de la classe minoritaire (churners)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📈 Ratio de déséquilibre : {scale_pos_weight:.2f} (appliqué à la classe positive)")

        # ==========================================================
        # ÉTAPE 5 : ENTRAÎNEMENT DU MODÈLE
        # ==========================================================
        print("🤖 Entraînement du modèle XGBoost...")

        model = XGBClassifier(
            n_estimators=301,
            learning_rate=0.034,
            max_depth=7,
            subsample=0.95,
            colsample_bytree=0.98,
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        )

        # Mesure du temps d'entraînement (performance)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Log du temps d'entraînement dans MLflow (métrique)
        mlflow.log_metric("train_time", train_time)
        print(f"✅ Modèle entraîné en {train_time:.2f} secondes")

        # ==========================================================
        # ÉTAPE 6 : ÉVALUATION
        # ==========================================================
        print("📊 Évaluation des performances...")

        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]

        # Application du seuil (plus bas = recall ↑ / precision ↓)
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1

        # Log du temps d'inférence dans MLflow (métrique)
        mlflow.log_metric("pred_time", pred_time)

        # Calcul des métriques
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, proba)

        # ==========================================================
        # LOG DES MÉTRIQUES DANS MLFLOW
        # ==========================================================
        # Ces métriques permettront de comparer plusieurs runs dans l’UI MLflow.
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"🎯 Performances :")
        print(f"   Precision : {precision:.3f} | Recall : {recall:.3f}")
        print(f"   F1 Score : {f1:.3f} | ROC AUC : {roc_auc:.3f}")

        # ==========================================================
        # ÉTAPE 7 : SAUVEGARDE DU MODÈLE (MLFLOW)
        # ==========================================================
        print("💾 Sauvegarde du modèle dans MLflow...")

        # mlflow.sklearn.log_model :
        # - sérialise le modèle
        # - crée un dossier d'artefacts "model/"
        # - permet ensuite :
        #   - de récupérer le modèle depuis MLflow
        #   - de servir le modèle via une API
        #   - de l'enregistrer en Model Registry (si activé)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )
        print("✅ Modèle sauvegardé dans MLflow (artefact)")

        # ==========================================================
        # RÉSUMÉ FINAL
        # ==========================================================
        print(f"\n⏱️  Résumé performance :")
        print(f"   Temps entraînement : {train_time:.2f}s")
        print(f"   Temps inférence    : {pred_time:.4f}s")
        print(f"   Samples / seconde  : {len(X_test)/pred_time:.0f}")

        print(f"\n📈 Rapport détaillé :")
        print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exécuter le pipeline churn avec XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="chemin vers le CSV (ex: data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="surcharge l'URI MLflow, sinon utilise project_root/mlruns")

    args = p.parse_args()
    main(args)

"""
# Exemple d'exécution du pipeline :

python scripts/run_pipeline.py \
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn

"""
