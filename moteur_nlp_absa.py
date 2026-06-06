import pandas as pd
import sqlite3
import re
import nltk
from nltk.corpus import stopwords
import sys, io
from tqdm import tqdm

# Téléchargement des mots vides en anglais (à exécuter une fois)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Fix de l'encodage de la console sous Windows
# Windows utilise parfois un encodage (comme CP1252) qui génère des erreurs lors de l'affichage
# de caractères accentués ou d'émojis UTF-8. Cette ligne force l'utilisation de l'UTF-8.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch
from transformers import pipeline

# =====================================================================
# CONFIGURATION ET CHARGEMENT DU MODÈLE SUR GPU
# =====================================================================
LIMITE_AVIS = 3000  

# Détection de la carte graphique (GPU) compatible CUDA.
device = 0 if torch.cuda.is_available() else -1
print(f"Utilisation du périphérique : {'GPU (GTX 1650)' if device == 0 else 'CPU'}")

print("Chargement du modèle de sentiment...")
# Chargement du modèle affiné (fine-tuned) précédemment sauvegardé dans le dossier './fine_tuned_distilbert'
# Nous utilisons la pipeline "sentiment-analysis" de Hugging Face pour simplifier les prédictions.
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="./fine_tuned_distilbert", 
    tokenizer="./fine_tuned_distilbert", 
    device=device
)

# =====================================================================
# DICTIONNAIRE DES ASPECTS 
# =====================================================================
# Cette structure associe chaque Aspect (la catégorie) à une liste de mots-clés (lexique).
# Si l'un de ces mots-clés apparaît dans un segment d'avis, ce segment sera associé à l'aspect correspondant.
dictionnaire_aspects = {
    "Hardware_Design": ["screen", "battery", "charge", "charger", "button", "size", "weight", "case", "cover", "color", "shape", "fit", "grip", "protect", "sturdy", "durability", "design", "look", "stick", "built"],
    "Software_OS": ["siri", "app", "application", "software", "ios", "android", "bluetooth", "connect", "connection", "wifi", "sync", "bug", "lag", "crash", "interface", "system", "feature"],
    "Logistics_Package": ["shipping", "delivery", "delivered", "shipped", "package", "packaging", "box", "arrive", "arrived", "late", "fast", "damage", "broken", "scratched"],
    "Price_Value": ["price", "cheap", "expensive", "cost", "value", "worth", "deal", "affordable", "refund", "return", "money", "buy", "purchase"]
}

# =====================================================================
# CHARGEMENT DES DONNÉES DEPUIS LE FICHIER JSON
# =====================================================================
chemin_fichier = "Cell_Phones_and_Accessories_5.json"
print(f"Lecture du fichier {chemin_fichier}...")

# Chargement optimisé par lignes (JSON Lines format) via Pandas.
df_raw = pd.read_json(chemin_fichier, lines=True)

if LIMITE_AVIS:
    df_raw = df_raw.head(LIMITE_AVIS)
    print(f"⚠️ Mode Test activé : traitement limité aux {LIMITE_AVIS} premiers avis.")

# =====================================================================
# CRÉATION DES TABLES SQLITE (STRUCTURATION DU PROJET)
# =====================================================================
# SQLite est un moteur de base de données relationnelle léger et local.
print("Connexion à la base de données SQLite...")
conn = sqlite3.connect("electro_marjane.db")

# Table 1 : Métadonnées des Avis (Avis_Bruts)
# Dans le fichier JSON d'origine, le champ 'helpful' est une liste [utiles, totaux].
# Par exemple, [3, 5] signifie que 3 personnes sur 5 ont trouvé l'avis utile.
# Nous extrayons ces valeurs pour créer des colonnes numériques claires.
df_raw['votes_utiles'] = df_raw['helpful'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0)
df_raw['votes_totaux'] = df_raw['helpful'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else 0)

# Sélection et renommage des colonnes pour avoir une table SQL propre et structurée.
df_avis_bruts = df_raw[['reviewerID', 'asin', 'overall', 'votes_utiles', 'votes_totaux', 'reviewTime']].copy()
df_avis_bruts.columns = ['ID_Utilisateur', 'ID_Produit', 'Note_Etoiles', 'Votes_Utiles', 'Votes_Totaux', 'Date_Avis']

# Sauvegarde de la table 1
# if_exists="replace" écrase la table existante si elle existe déjà.
df_avis_bruts.to_sql("Avis_Bruts", conn, if_exists="replace", index=False)
print("Table 'Avis_Bruts' sauvegardée.")

# =====================================================================
# TRAITEMENT NLP & ABSA (ANALYSE DE SENTIMENT PAR ASPECT)
# =====================================================================
print("Démarrage de l'analyse sémantique ABSA...")

# Remplacement des valeurs manquantes (NaN) par des chaînes de caractères vides
df_raw['reviewText'] = df_raw['reviewText'].fillna('')
df_raw['summary'] = df_raw['summary'].fillna('')
# Fusion du résumé et du texte pour une analyse complète.
df_raw['Texte_Complet'] = df_raw['summary'] + ". " + df_raw['reviewText']

lignes_absa = []

# tqdm permet d'afficher une jolie barre de progression dans la console durant le traitement.
for index, row in tqdm(df_raw.iterrows(), total=df_raw.shape[0], desc="Analyse ABSA en cours"):
    texte = str(row['Texte_Complet']).lower()
    # Nettoyage de base : supprime les caractères spéciaux indésirables tout en préservant
    # la ponctuation fondamentale (.,!?;\') car elle sert à la segmentation.
    texte_propre = re.sub(r"[^a-zA-Z0-9\s.,!?;\']", "", texte)
    
    # SEGMENTATION DES PHRASES : 
    # C'est la clé de l'analyse ABSA. Un avis complet peut contenir des sentiments opposés :
    # "The screen is great, but the shipping was slow."
    # Si nous analysions tout le texte d'un coup, le modèle mélangerait les sentiments.
    # En découpant sur la ponctuation et les conjonctions de coordination ('but', 'and'),
    # nous isolons les propositions indépendantes :
    # Segment 1 : "the screen is great" -> Aspect : Hardware_Design -> Positif
    # Segment 2 : "the shipping was slow" -> Aspect : Logistics_Package -> Négatif
    segments = re.split(r'[.,;!]|\bbut\b|\band\b', texte_propre)
    
    for segment in segments:
        segment = segment.strip()
        # On ignore les segments trop courts (bruit ou espaces vides)
        if len(segment) < 3:
            continue
            
        # === NOUVEAU : UTILISATION DE NLTK ICI ===
        # Suppression des Stopwords (mots vides) avec NLTK
        mots = segment.split()
        mots_filtres = [mot for mot in mots if mot.lower() not in stop_words]
        segment_propre = " ".join(mots_filtres)
        
        # On ignore si le segment devient vide après le nettoyage NLTK
        if len(segment_propre) < 3:
            continue
        # =========================================
            
        # DÉTECTION DE L'ASPECT (Approche basée sur le lexique/règles sur le segment nettoyé)
        aspect_detecte = None
        for aspect, mots_cles in dictionnaire_aspects.items():
            # Si l'un des mots-clés de l'aspect est présent dans le segment nettoyé
            if any(mot in segment_propre for mot in mots_cles):
                aspect_detecte = aspect
                break  # On associe le segment au premier aspect trouvé
                
        # ANALYSE DE SENTIMENT SUR LE SEGMENT (Approche par Deep Learning)
        # Si un aspect a été détecté dans ce segment, on passe le segment nettoyé au modèle de Deep Learning.
        if aspect_detecte:
            # segment_propre[:512] : On tronque à 512 caractères maximum pour respecter la limite
            # de taille d'entrée du modèle DistilBERT et éviter des erreurs de dimension.
            prediction = sentiment_analyzer(segment_propre[:512])[0]
            
            # Mappage des sorties brutes du modèle fine-tuné DistilBERT.
            # Le modèle a été entraîné à prédire LABEL_0, LABEL_1, LABEL_2.
            # Nous les convertissons en termes compréhensibles pour l'humain et le professeur.
            mappage_nouveau = {
                "LABEL_0": "Négatif",
                "LABEL_1": "Neutre",
                "LABEL_2": "Positif"
            }
            sentiment = mappage_nouveau[prediction['label']]
                
            # Enregistrement des résultats structurés pour ce segment d'avis.
            lignes_absa.append({
                "ID_Avis": row['reviewerID'],
                "Segment_Texte": segment_propre,
                "Aspect_Detecte": aspect_detecte,
                "Sentiment_Aspect": sentiment,
                "Confiance_IA": round(prediction['score'], 2)  # Probabilité associée à la prédiction
            })

# Table 2 : Résultats de l'analyse ABSA
# Nous convertissons la liste de dictionnaires en DataFrame Pandas, puis l'écrivons dans la base SQLite.
df_absa = pd.DataFrame(lignes_absa)
df_absa.to_sql("Analyses_ABSA", conn, if_exists="replace", index=False)
print("Table 'Analyses_ABSA' sauvegardée.")

# Fermeture propre de la connexion à la base de données.
conn.close()
print("\n🎉 BASE DE DONNÉES 'electro_marjane.db' GÉNÉRÉE AVEC SUCCÈS !")