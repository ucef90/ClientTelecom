import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna

print("=== Phase 2 : Modélisation avec XGBoost ===")

# ==========================================================
# 1) Chargement du dataset traité (prêt pour le ML)
# ==========================================================
df = pd.read_csv("data/processed/telco_churn_processed.csv")

# ==========================================================
# 2) Sécurisation de la cible : Churn doit être numérique (0/1)
# ==========================================================
# Si la colonne cible est encore en texte ("Yes"/"No"), on la convertit
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

# Vérifications de cohérence (sanity checks)
assert df["Churn"].isna().sum() == 0, "Churn contient des NaN"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn n'est pas encodée en 0/1"

# ==========================================================
# 3) Séparation features (X) / cible (y)
# ==========================================================
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ==========================================================
# 4) Split train/test (stratifié)
# ==========================================================
# stratify=y permet de conserver la même proportion de churners
# dans train et test (important en cas de dataset déséquilibré)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================================================
# 5) Seuil de décision (threshold)
# ==========================================================
# Ici on ne se contente pas de model.predict() (classe directe),
# on utilise predict_proba() puis un seuil personnalisé.
#
# Pourquoi ?
# - Le churn est souvent un problème déséquilibré
# - On veut souvent augmenter le recall (détecter + de churners)
# - Un seuil plus bas → recall ↑ mais precision ↓
THRESHOLD = 0.4


# ==========================================================
# 6) Objectif Optuna : optimiser les hyperparamètres
# ==========================================================
# Optuna va appeler objective(trial) plusieurs fois.
# À chaque trial, Optuna propose un jeu d'hyperparamètres,
# entraîne un modèle, puis évalue une métrique (ici : recall).
def objective(trial):

    # ------------------------------------------------------
    # 6.1 Espace de recherche des hyperparamètres XGBoost
    # ------------------------------------------------------
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        # Paramètres de robustesse / régularisation
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),

        # Paramètres de reproductibilité / performance
        "random_state": 42,
        "n_jobs": -1,

        # Gestion du déséquilibre : donne plus de poids à la classe churn (1)
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),

        # Métrique interne XGBoost (pour stabilité de l'entraînement)
        "eval_metric": "logloss",
    }

    # ------------------------------------------------------
    # 6.2 Entraînement du modèle sur train
    # ------------------------------------------------------
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # ------------------------------------------------------
    # 6.3 Évaluation sur test avec seuil personnalisé
    # ------------------------------------------------------
    # predict_proba renvoie une probabilité de churn.
    # On transforme en classe via THRESHOLD.
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    # ------------------------------------------------------
    # 6.4 Métrique optimisée : recall
    # ------------------------------------------------------
    # recall = capacité à détecter les churners (classe 1)
    from sklearn.metrics import recall_score
    return recall_score(y_test, y_pred, pos_label=1)


# ==========================================================
# 7) Lancement de l’étude Optuna
# ==========================================================
# direction="maximize" car on cherche à maximiser le recall
study = optuna.create_study(direction="maximize")

# n_trials=30 → Optuna teste 30 configurations différentes
study.optimize(objective, n_trials=30)

# ==========================================================
# 8) Résultats de l’optimisation
# ==========================================================
print("Best Params:", study.best_params)
print("Best Recall:", study.best_value)


# =====================================================================
# EXPLICATION GLOBALE – PHASE 2 : TUNING XGBOOST AVEC OPTUNA (MLOps)
# =====================================================================
#
# Objectif de ce script :
# - Optimiser automatiquement les hyperparamètres d’un modèle XGBoost
# - Maximiser une métrique métier adaptée au churn : le recall
#
# Pourquoi le recall ?
# - Dans le churn, rater un churner (FN) coûte souvent cher
# - On préfère donc attraper un maximum de churners (recall élevé),
#   même si cela augmente les faux positifs (precision plus faible)
#
# Fonctionnement d’Optuna ici :
# 1) create_study() crée une "étude" d’optimisation
# 2) optimize() exécute plusieurs trials
# 3) chaque trial :
#    - choisit des hyperparamètres via suggest_*
#    - entraîne un modèle
#    - calcule le recall
# 4) best_params contient les meilleurs hyperparamètres trouvés
#
# Rôle du THRESHOLD :
# - On prédit une probabilité, pas directement une classe
# - Le seuil permet d’ajuster le compromis recall/precision
# - En production, ce seuil doit souvent être calibré métier
#
# Limite (à connaître en formation/jury) :
# - Ici l’évaluation est faite sur le même X_test / y_test
#   à chaque trial (risque de sur-optimisation sur ce test).
# - En industrialisation avancée, on préfère souvent :
#   - cross-validation
#   - ou un jeu de validation séparé
