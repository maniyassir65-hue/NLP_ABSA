import pandas as pd
import sqlite3
import re
import nltk
from nltk.corpus import stopwords
import sys, io
from tqdm import tqdm

# Configuration linguistique (NLTK)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Force l'encodage UTF-8 pour la console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch
from transformers import pipeline

# Configuration et modèle
LIMITE_AVIS = 3000  
device = 0 if torch.cuda.is_available() else -1
print(f"Périphérique : {'GPU (GTX 1650)' if device == 0 else 'CPU'}")

print("Chargement du modèle de sentiment...")
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="./fine_tuned_distilbert", 
    tokenizer="./fine_tuned_distilbert", 
    device=device
)

# Lexique pour la classification d'aspects (Ontologie)
dictionnaire_aspects = {
    "Hardware_Design": ["screen", "battery", "charge", "charger", "button", "size", "weight", "case", "cover", "color", "shape", "fit", "grip", "protect", "sturdy", "durability", "design", "look", "stick", "built"],
    "Software_OS": ["siri", "app", "application", "software", "ios", "android", "bluetooth", "connect", "connection", "wifi", "sync", "bug", "lag", "crash", "interface", "system", "feature"],
    "Logistics_Package": ["shipping", "delivery", "delivered", "shipped", "package", "packaging", "box", "arrive", "arrived", "late", "fast", "damage", "broken", "scratched"],
    "Price_Value": ["price", "cheap", "expensive", "cost", "value", "worth", "deal", "affordable", "refund", "return", "money", "buy", "purchase"]
}

# Chargement du dataset brut
chemin_fichier = "Cell_Phones_and_Accessories_5.json"
print(f"Lecture de {chemin_fichier}...")
df_raw = pd.read_json(chemin_fichier, lines=True)

if LIMITE_AVIS:
    df_raw = df_raw.head(LIMITE_AVIS)
    print(f"⚠️ Mode Test actif : limité aux {LIMITE_AVIS} premiers avis.")

# Structuration dans SQLite
print("Connexion à la base SQLite...")
conn = sqlite3.connect("electro_marjane.db")

# Table 1 : Métadonnées des Avis
df_raw['votes_utiles'] = df_raw['helpful'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0)
df_raw['votes_totaux'] = df_raw['helpful'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else 0)

df_avis_bruts = df_raw[['reviewerID', 'asin', 'overall', 'votes_utiles', 'votes_totaux', 'reviewTime']].copy()
df_avis_bruts.columns = ['ID_Utilisateur', 'ID_Produit', 'Note_Etoiles', 'Votes_Utiles', 'Votes_Totaux', 'Date_Avis']

df_avis_bruts.to_sql("Avis_Bruts", conn, if_exists="replace", index=False)
print("Table 'Avis_Bruts' enregistrée.")

# Pipeline de traitement NLP & ABSA
print("Traitement ABSA en cours...")
df_raw['reviewText'] = df_raw['reviewText'].fillna('')
df_raw['summary'] = df_raw['summary'].fillna('')
df_raw['Texte_Complet'] = df_raw['summary'] + ". " + df_raw['reviewText']

lignes_absa = []

for index, row in tqdm(df_raw.iterrows(), total=df_raw.shape[0], desc="ABSA"):
    texte = str(row['Texte_Complet']).lower()
    texte_propre = re.sub(r"[^a-zA-Z0-9\s.,!?;\']", "", texte)
    
    # Segmentation en clauses sémantiques distinctes
    segments = re.split(r'[.,;!]|\bbut\b|\band\b', texte_propre)
    
    for segment in segments:
        segment = segment.strip()
        if len(segment) < 3:
            continue
            
        # Filtrage linguistique (mots vides)
        mots = segment.split()
        mots_filtres = [mot for mot in mots if mot.lower() not in stop_words]
        segment_propre = " ".join(mots_filtres)
        
        if len(segment_propre) < 3:
            continue
            
        # Détection d'aspect (lexicale)
        aspect_detecte = None
        for aspect, mots_cles in dictionnaire_aspects.items():
            if any(mot in segment_propre for mot in mots_cles):
                aspect_detecte = aspect
                break
                
        # Inférence de sentiment via le modèle fine-tuné
        if aspect_detecte:
            prediction = sentiment_analyzer(segment_propre[:512])[0]
            
            mappage_nouveau = {
                "LABEL_0": "Négatif",
                "LABEL_1": "Neutre",
                "LABEL_2": "Positif"
            }
            sentiment = mappage_nouveau[prediction['label']]
                
            lignes_absa.append({
                "ID_Avis": row['reviewerID'],
                "Segment_Texte": segment_propre,
                "Aspect_Detecte": aspect_detecte,
                "Sentiment_Aspect": sentiment,
                "Confiance_IA": round(prediction['score'], 2)
            })

# Table 2 : Résultats de l'analyse ABSA
df_absa = pd.DataFrame(lignes_absa)
df_absa.to_sql("Analyses_ABSA", conn, if_exists="replace", index=False)
print("Table 'Analyses_ABSA' enregistrée.")

conn.close()
print("\n🎉 Base de données générée avec succès !")