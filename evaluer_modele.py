import sqlite3
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =====================================================================
# 1. CONNEXION À LA BASE DE DONNÉES LOCALES
# =====================================================================
# Nous nous connectons à la base de données générée par le moteur ABSA.
conn = sqlite3.connect("electro_marjane.db")

# =====================================================================
# 2. JOINTURE SQL ENTRE MÉTADONNÉES ET PRÉDICTIONS NLP
# =====================================================================
# Pour évaluer les performances de notre IA, nous devons comparer ses prédictions
# avec la "Vérité Terrain" (l'avis de l'utilisateur matérialisé par sa note étoilée).
# Nous effectuons une jointure SQL (JOIN) entre la table des avis bruts et celle
# des sentiments par aspect, en liant le reviewer ID/ID Utilisateur.
query = """
SELECT b.Note_Etoiles, a.Sentiment_Aspect
FROM Avis_Bruts b
JOIN Analyses_ABSA a ON b.ID_Utilisateur = a.ID_Avis
"""
df_eval = pd.read_sql_query(query, conn)
conn.close()

# =====================================================================
# 3. MAPPING DU SENTIMENT RÉEL (VÉRITÉ TERRAIN)
# =====================================================================
# Le sentiment prédit par notre modèle est textuel ("Positif", "Neutre", "Négatif").
# La note étoilée est numérique (de 1 à 5). Pour pouvoir les comparer,
# nous convertissons les notes étoilées en sentiments selon la même logique :
# - 4 ou 5 étoiles -> "Positif"
# - 3 étoiles -> "Neutre"
# - 1 ou 2 étoiles -> "Négatif"
def convertir_etoiles_en_sentiment(stars):
    if stars >= 4:
        return "Positif"
    elif stars <= 2:
        return "Négatif"
    else:
        return "Neutre"

df_eval['Sentiment_Reel'] = df_eval['Note_Etoiles'].apply(convertir_etoiles_en_sentiment)

# Extraction des deux vecteurs : les valeurs réelles (y_true) et les valeurs prédites par l'IA (y_pred)
y_true = df_eval['Sentiment_Reel']
y_pred = df_eval['Sentiment_Aspect']

# =====================================================================
# 4. CALCUL DES MÉTRIQUES SCIENTIFIQUES D'ÉVALUATION
# =====================================================================
print("==================================================")
print("     RAPPORT D'ÉVALUATION SCIENTIFIQUE DU MODÈLE   ")
print("==================================================")

# L'Accuracy (Exactitude globale) : 
# Pourcentage total de prédictions correctes de l'IA toutes classes confondues.
# Formule : (Vrais Positifs + Vrais Négatifs) / Total des prédictions.
accuracy = accuracy_score(y_true, y_pred)
print(f"Exactitude Globale (Accuracy) : {accuracy:.2%}\n")

# Rapport de classification détaillé de Scikit-Learn :
# Ce rapport affiche pour chaque sentiment (Positif, Neutre, Négatif) :
# - la Précision : parmis tous les segments prédits comme [Sentiment], combien le sont réellement ?
#   (Formule : Vrais Positifs / (Vrais Positifs + Faux Positifs))
# - le Rappel (Recall) : parmis tous les segments réellement de classe [Sentiment], combien l'IA a-t-elle réussi à trouver ?
#   (Formule : Vrais Positifs / (Vrais Positifs + Faux Négatifs))
# - le F1-Score : la moyenne harmonique de la précision et du rappel. C'est l'indicateur le plus robuste en cas de classes déséquilibrées.
# - le Support : le nombre total d'exemples réels de chaque classe présents dans le jeu de test.
print("Rapport de classification détaillé :")
print(classification_report(y_true, y_pred))

# Matrice de Confusion (Confusion Matrix) :
# Tableau récapitulatif permettant de voir exactement quels sentiments ont été confondus.
# Les lignes représentent la réalité (Vérité terrain) et les colonnes les prédictions de l'IA.
# La diagonale principale indique les classifications correctes (ex: Réel_Positif prédit comme Positif).
# Les valeurs hors diagonale montrent les erreurs (ex: Réel_Négatif prédit comme Positif).
print("Matrice de Confusion :")
labels = ["Négatif", "Neutre", "Positif"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
df_cm = pd.DataFrame(cm, index=[f"Réel_{l}" for l in labels], columns=[f"Prédit_{l}" for l in labels])
print(df_cm)
print("==================================================")