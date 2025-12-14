import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


def tune_model(X, y):
    """
    Optimisation des hyperparamètres d'un modèle XGBoost à l'aide d'Optuna.

    Cette fonction utilise Optuna pour rechercher automatiquement
    la meilleure combinaison d'hyperparamètres afin de maximiser
    une métrique cible (ici le recall).

    Objectif MLOps :
    - Améliorer les performances du modèle
    - Automatiser la recherche d'hyperparamètres
    - Garantir une optimisation reproductible et traçable
    """

    # ==========================================================
    # FONCTION OBJECTIF (OPTUNA)
    # ==========================================================
    # Cette fonction est appelée plusieurs fois par Optuna.
    # À chaque appel, Optuna propose un nouvel ensemble
    # d'hyperparamètres à tester.
    def objective(trial):

        # ------------------------------------------------------
        # 1. Définition de l'espace de recherche
        # ------------------------------------------------------
        # Optuna explore ces plages de valeurs de manière intelligente
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }

        # Initialisation du modèle avec les paramètres proposés
        model = XGBClassifier(**params)

        # ------------------------------------------------------
        # 2. Évaluation par validation croisée
        # ------------------------------------------------------
        # cross_val_score :
        # - entraîne le modèle sur plusieurs folds
        # - réduit le risque d'overfitting
        # - fournit une estimation plus robuste de la performance
        scores = cross_val_score(
            model,
            X,
            y,
            cv=3,
            scoring="recall"
        )

        # Optuna cherche à maximiser la moyenne du recall
        return scores.mean()

    # ==========================================================
    # LANCEMENT DE L'ÉTUDE OPTUNA
    # ==========================================================
    # direction="maximize" car on souhaite maximiser le recall
    study = optuna.create_study(direction="maximize")

    # Lancement de l'optimisation :
    # - objective : fonction à optimiser
    # - n_trials : nombre d'expériences (essais)
    study.optimize(
        objective,
        n_trials=20
    )

    # ==========================================================
    # RÉSULTATS DE L'OPTIMISATION
    # ==========================================================
    # study.best_params contient la meilleure configuration trouvée
    print("Best Params:", study.best_params)

    return study.best_params


# =====================================================================
# EXPLICATION GLOBALE – OPTUNA & HYPERPARAMETER TUNING (MLOps)
# =====================================================================
#
# Rôle de ce module :
# - Automatiser la recherche des hyperparamètres optimaux
# - Remplacer une recherche manuelle ou un GridSearch coûteux
#
# Fonctionnement d’Optuna :
#
# 1. Study
#    → représente une étude d'optimisation
#    → contient l'historique de tous les essais (trials)
#
# 2. Trial
#    → correspond à UNE tentative avec un jeu de paramètres
#    → Optuna choisit les valeurs à tester de manière adaptative
#
# 3. objective()
#    → fonction que l'on cherche à optimiser
#    → retourne une métrique scalaire (ici le recall)
#
# 4. Algorithme d’optimisation
#    → Optuna utilise des méthodes bayésiennes (TPE)
#    → exploration + exploitation intelligente de l’espace de recherche
#
# Choix de la métrique (recall) :
# - Le churn est un problème déséquilibré
# - Il est plus critique de détecter les clients à risque
# - Le recall est donc plus pertinent que l’accuracy seule
#
# Bonnes pratiques MLOps respectées :
# - Validation croisée pour robustesse
# - Random state fixé pour reproductibilité
# - Nombre d’essais contrôlé (n_trials)
#
# Intégration typique dans le pipeline :
# - build_features
# - tune_model (ce module)
# - train_model avec best_params
# - log des résultats via MLflow
