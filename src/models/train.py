import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


def train_model(df: pd.DataFrame, target_col: str):
    """
    Entraîne un modèle XGBoost et journalise l'expérience avec MLflow.

    Cette fonction couvre l'étape d'entraînement du pipeline ML et assure
    la traçabilité complète de l'expérience :
    - paramètres du modèle
    - métriques de performance
    - artefact modèle
    - données utilisées pour l'entraînement

    Objectif MLOps :
    Garantir la reproductibilité, la comparabilité et l'auditabilité
    des expériences de Machine Learning.
    """

    # ==========================================================
    # 1. Séparation des features (X) et de la cible (y)
    # ==========================================================
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ==========================================================
    # 2. Découpage du dataset (train / test)
    # ==========================================================
    # test_size=0.2 → 80% entraînement / 20% évaluation
    # random_state fixé pour reproductibilité
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # ==========================================================
    # 3. Initialisation du modèle XGBoost
    # ==========================================================
    model = XGBClassifier(
        n_estimators=300,      # Nombre d'arbres
        learning_rate=0.1,     # Taux d'apprentissage
        max_depth=6,           # Profondeur maximale des arbres
        random_state=42,       # Reproductibilité
        n_jobs=-1,             # Utilisation de tous les cœurs CPU
        eval_metric="logloss"  # Métrique interne XGBoost
    )

    # ==========================================================
    # 4. DÉMARRAGE D'UN RUN MLFLOW
    # ==========================================================
    # mlflow.start_run() ouvre un "run" qui regroupe :
    # - paramètres
    # - métriques
    # - artefacts
    # - métadonnées
    #
    # Chaque appel correspond à UNE expérience traçable dans l'UI MLflow
    with mlflow.start_run():

        # ------------------------------------------------------
        # 4.1 Entraînement du modèle
        # ------------------------------------------------------
        model.fit(X_train, y_train)

        # Prédictions sur le jeu de test
        preds = model.predict(X_test)

        # Calcul des métriques d'évaluation
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        # ------------------------------------------------------
        # 4.2 Journalisation des paramètres (MLflow)
        # ------------------------------------------------------
        # Les paramètres permettent de comparer différentes expériences
        mlflow.log_param("n_estimators", 300)

        # ------------------------------------------------------
        # 4.3 Journalisation des métriques (MLflow)
        # ------------------------------------------------------
        # Les métriques sont affichées sous forme de courbes / tableaux
        # dans l'interface MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)

        # ------------------------------------------------------
        # 4.4 Journalisation du modèle (artefact)
        # ------------------------------------------------------
        # Le modèle est sauvegardé comme artefact MLflow
        # → réutilisable pour :
        #    - le serving
        #    - le model registry
        #    - le déploiement
        mlflow.xgboost.log_model(model, "model")

        # ------------------------------------------------------
        # 4.5 Journalisation des données d'entraînement
        # ------------------------------------------------------
        # Permet d'associer explicitement le dataset au run MLflow
        # (traçabilité des données = point clé MLOps)
        train_ds = mlflow.data.from_pandas(
            df,
            source="training_data"
        )
        mlflow.log_input(train_ds, context="training")

        # ------------------------------------------------------
        # 4.6 Résumé console
        # ------------------------------------------------------
        print(f"Modèle entraîné. Accuracy: {acc:.4f}, Recall: {rec:.4f}")


# =====================================================================
# EXPLICATION GLOBALE – ENTRAÎNEMENT & MLFLOW (MLOps)
# =====================================================================
#
# Rôle de ce module :
# - Entraîner un modèle XGBoost
# - Tracer l'expérience complète avec MLflow
#
# Fonctionnement de MLflow dans ce code :
#
# 1. start_run()
#    → crée une expérience unique (run) identifiable dans l'UI MLflow
#
# 2. log_param()
#    → enregistre les hyperparamètres du modèle
#    → permet de comparer plusieurs entraînements
#
# 3. log_metric()
#    → stocke les performances du modèle (accuracy, recall, etc.)
#    → visualisables sous forme de graphiques
#
# 4. log_model()
#    → sauvegarde le modèle entraîné comme artefact
#    → utilisable pour le serving ou le Model Registry
#
# 5. log_input()
#    → lie explicitement les données utilisées au run
#    → essentiel pour l'auditabilité et la gouvernance des modèles
#
# Pourquoi c'est critique en industrialisation :
# - Reproductibilité des résultats
# - Comparaison d'expériences
# - Traçabilité des données et des modèles
# - Base du déploiement continu (CI/CD ML)
#
# Ce module s'intègre typiquement après :
# - load_data
# - preprocess_data
# - build_features
#
# Et avant :
# - évaluation avancée
# - enregistrement en Model Registry
# - déploiement API / Docker / AWS
