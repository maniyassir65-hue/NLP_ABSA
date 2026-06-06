import sqlite3
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Connexion à la base de données locale
conn = sqlite3.connect("electro_marjane.db")

# Récupération des notes réelles (Avis_Bruts) et des prédictions (Analyses_ABSA)
query = """
SELECT b.Note_Etoiles, a.Sentiment_Aspect
FROM Avis_Bruts b
JOIN Analyses_ABSA a ON b.ID_Utilisateur = a.ID_Avis
"""
df_eval = pd.read_sql_query(query, conn)
conn.close()

# Conversion des notes étoilées en sentiments réels (Vérité terrain)
def convertir_etoiles_en_sentiment(stars):
    if stars >= 4:
        return "Positif"
    elif stars <= 2:
        return "Négatif"
    else:
        return "Neutre"

df_eval['Sentiment_Reel'] = df_eval['Note_Etoiles'].apply(convertir_etoiles_en_sentiment)

y_true = df_eval['Sentiment_Reel']
y_pred = df_eval['Sentiment_Aspect']

# Calcul et affichage des métriques scientifiques
print("==================================================")
print("     RAPPORT D'ÉVALUATION SCIENTIFIQUE DU MODÈLE   ")
print("==================================================")

# Accuracy (Exactitude globale)
accuracy = accuracy_score(y_true, y_pred)
print(f"Exactitude Globale (Accuracy) : {accuracy:.2%}\n")

# Rapport détaillé de Scikit-Learn (Précision, Rappel, F1-Score)
print("Rapport de classification détaillé :")
print(classification_report(y_true, y_pred))

# Matrice de confusion
print("Matrice de Confusion :")
labels = ["Négatif", "Neutre", "Positif"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
df_cm = pd.DataFrame(cm, index=[f"Réel_{l}" for l in labels], columns=[f"Prédit_{l}" for l in labels])
print(df_cm)
print("==================================================")