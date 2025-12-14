from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances d’un modèle de classification
    sur un jeu de données de test.

    Cette fonction fournit une analyse détaillée des résultats
    du modèle à travers des métriques standard utilisées en
    Machine Learning pour les problèmes de classification.
    """

    # ==========================================================
    # 1. Génération des prédictions
    # ==========================================================
    # Le modèle prédit la classe (0 ou 1) pour chaque observation
    preds = model.predict(X_test)

    # ==========================================================
    # 2. Rapport de classification
    # ==========================================================
    # classification_report fournit :
    # - precision
    # - recall
    # - f1-score
    # - support
    #
    # Ces métriques permettent d'évaluer finement les performances
    # du modèle, en particulier sur des datasets déséquilibrés
    print("Classification Report:\n", classification_report(y_test, preds))

    # ==========================================================
    # 3. Matrice de confusion
    # ==========================================================
    # La matrice de confusion permet d'analyser :
    # - vrais positifs (TP)
    # - faux positifs (FP)
    # - vrais négatifs (TN)
    # - faux négatifs (FN)
    #
    # Elle est essentielle pour comprendre les erreurs du modèle
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))


# =====================================================================
# EXPLICATION GLOBALE – ÉVALUATION DU MODÈLE (MLOps)
# =====================================================================
#
# Objectif de ce module :
# - Évaluer objectivement la performance du modèle entraîné
# - Comprendre les types d’erreurs commises
#
# Pourquoi cette étape est critique :
# - L’accuracy seule est insuffisante pour les problèmes déséquilibrés
# - Le churn nécessite une attention particulière au recall
#
# Interprétation des métriques :
#
# - Precision :
#   Proportion de prédictions positives correctes
#
# - Recall :
#   Capacité du modèle à détecter les clients churners
#   (métrique métier clé pour le churn)
#
# - F1-score :
#   Compromis entre precision et recall
#
# - Matrice de confusion :
#   Donne une vision détaillée des erreurs du modèle
#
# Intégration dans le pipeline :
# - Après train_model
# - Avant validation métier finale ou déploiement
#
# Bonnes pratiques MLOps :
# - Séparation claire entraînement / évaluation
# - Métriques interprétables par des non-techniciens
# - Base pour le monitoring post-déploiement
