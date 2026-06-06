import pandas as pd
import torch
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# =====================================================================
# 1. PARAMÈTRES ET CONFIGURATION DU GPU
# =====================================================================
# Nous choisissons "distilbert-base-uncased" comme modèle de base.
# DistilBERT est une version distillée (compressée) de BERT. Elle conserve environ 97%
# des performances de BERT tout en étant 40% plus petite et 60% plus rapide.
# C'est le choix idéal pour un entraînement sur du matériel grand public (comme une GTX 1650).
NOM_MODELE = "distilbert-base-uncased"  

# Taille de l'échantillon pour l'entraînement. Traiter les millions d'avis prendrait
# trop de temps et de mémoire, nous limitons donc ici à 30 000 exemples pour le projet.
TAILLE_ECHANTILLON = 30000            

# Sélection automatique du GPU s'il est disponible (via CUDA), sinon repli sur le CPU.
# Le GPU accélère considérablement l'entraînement des réseaux de neurones (Deep Learning).
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"L'entraînement s'exécutera sur : {device.upper()}")

# =====================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES (AMAZON JSON)
# =====================================================================
print("Chargement des données pour l'entraînement...")
# Lecture du fichier JSON contenant les avis. 'lines=True' signifie que chaque ligne du fichier
# est un objet JSON indépendant (format JSON Lines).
if TAILLE_ECHANTILLON:
    df = pd.read_json("Cell_Phones_and_Accessories_5.json", lines=True).head(TAILLE_ECHANTILLON)
else:
    df = pd.read_json("Cell_Phones_and_Accessories_5.json", lines=True)

# Concaténation du Titre ('summary') et du Corps de l'avis ('reviewText')
# Cela fournit plus de contexte textuel au modèle pour sa prédiction de sentiment.
df['Texte'] = df['summary'].fillna('') + ". " + df['reviewText'].fillna('')

# Mappage des étoiles (notes de 1 à 5) en 3 classes numériques :
# Les modèles de classification sous PyTorch ont besoin d'étiquettes entières (indices de classes)
# allant de 0 à (N-1), où N est le nombre de classes (ici 3 classes : 0, 1, 2).
def convertir_etoiles_en_label(stars):
    if stars >= 4:
        return 2  # Sentiment Positif (Note de 4 ou 5)
    elif stars <= 2:
        return 0  # Sentiment Négatif (Note de 1 ou 2)
    else:
        return 1  # Sentiment Neutre (Note de 3)

df['label'] = df['overall'].apply(convertir_etoiles_en_label)

# Séparation des données : 80% pour l'Entraînement / 20% pour la Validation
# L'ensemble de validation permet de mesurer la capacité de généralisation du modèle
# et de surveiller le surapprentissage (overfitting) pendant l'entraînement.
textes_train, textes_val, labels_train, labels_val = train_test_split(
    df['Texte'].tolist(), 
    df['label'].tolist(), 
    test_size=0.2,       # 20% pour l'évaluation
    random_state=42      # Fixe la graine aléatoire pour avoir des résultats reproductibles
)

# =====================================================================
# 3. TOKENISATION (PRÉPARATION DU TEXTE POUR BERT)
# =====================================================================
print("Initialisation du Tokenizer...")
# Le Tokenizer convertit les phrases brutes en une suite de jetons (tokens) ou sous-mots,
# puis mappe ces tokens à leurs identifiants numériques (IDs) dans le dictionnaire de BERT.
tokenizer = AutoTokenizer.from_pretrained(NOM_MODELE)

# Tokenisation des ensembles d'entraînement et de validation.
# - truncation=True : Coupe le texte si sa longueur dépasse max_length.
# - padding=True : Ajoute des tokens de remplissage (PAD) pour que toutes les séquences
#   du lot aient la même longueur (nécessaire pour le calcul matriciel sur GPU).
# - max_length=128 : Limite la longueur à 128 jetons. C'est un compromis idéal qui
#   évite la saturation de la VRAM (mémoire vidéo) du GPU GTX 1650.
print("Tokenisation des textes (Train & Val)...")
train_encodings = tokenizer(textes_train, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(textes_val, truncation=True, padding=True, max_length=128)

# =====================================================================
# 4. CRÉATION DU DATASET PYTORCH (Obligatoire pour Hugging Face Trainer)
# =====================================================================
# PyTorch nécessite que les données d'entraînement soient encapsulées dans un objet Dataset.
# Cette classe hérite de torch.utils.data.Dataset et définit l'accès aux éléments.
class AmazonReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # Retourne un dictionnaire contenant les tenseurs (Tensors) PyTorch pour un exemple donné (idx).
    # Ces tenseurs incluent les 'input_ids' (indices des mots) et l'attention_mask' (pour ignorer le padding).
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    # Retourne le nombre total d'exemples dans le jeu de données
    def __len__(self):
        return len(self.labels)

# Instanciation de nos objets datasets
dataset_train = AmazonReviewsDataset(train_encodings, labels_train)
dataset_val = AmazonReviewsDataset(val_encodings, labels_val)

# =====================================================================
# 5. CHARGEMENT DU MODÈLE ET AJUSTEMENT DES HYPERPARAMÈTRES (VRAM OPTIMISÉE)
# =====================================================================
print("Chargement du modèle de classification (3 classes)...")
# Chargement du modèle de classification DistilBERT pré-entraîné.
# num_labels=3 indique que la couche finale de classification aura 3 neurones de sortie
# correspondant à nos 3 classes (Négatif, Neutre, Positif).
model = AutoModelForSequenceClassification.from_pretrained(NOM_MODELE, num_labels=3)

# Configuration de l'entraînement optimisée pour carte graphique 4Go (ex: GTX 1650)
training_args = TrainingArguments(
    output_dir="./results",           # Répertoire de sauvegarde des points de contrôle (checkpoints)
    num_train_epochs=2,              # 2 époques (le modèle parcourt 2 fois l'intégralité du dataset d'entraînement)
    per_device_train_batch_size=4,   # Lot de 4 avis par étape. Réduit pour ne pas saturer les 4Go de VRAM.
    per_device_eval_batch_size=4,    # Lot pour la validation
    warmup_steps=100,                # Phase de montée en température du taux d'apprentissage (évite des gradients instables au début)
    weight_decay=0.01,               # Régularisation L2 pour limiter le surapprentissage
    logging_dir="./logs",            # Répertoire des logs pour le suivi (ex: TensorBoard)
    logging_steps=50,                # Journalisation toutes les 50 étapes d'apprentissage
    eval_strategy="epoch",           # Évaluation du modèle à la fin de chaque époque
    save_strategy="epoch",           # Sauvegarde du checkpoint à chaque époque
    fp16=True,                       # CALCULS EN Précision Mixte 16-BIT (Divise par 2 l'usage VRAM, accélère le calcul, indispensable sur GTX 1650 !)
    load_best_model_at_end=True,     # Charger la meilleure version du modèle (selon la métrique) à la fin
    metric_for_best_model="accuracy" # Utiliser l'Exactitude globale (Accuracy) comme critère de sélection
)

# Fonction de calcul des métriques durant l'évaluation
# Convertit les logits (sorties brutes du modèle) en prédictions d'indices via argmax,
# puis calcule le taux de bonnes réponses (Accuracy) par rapport aux vrais labels.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# =====================================================================
# 6. LANCEMENT DE L'ENTRAÎNEMENT (FINE-TUNING)
# =====================================================================
# Hugging Face Trainer encapsule toute la boucle d'apprentissage PyTorch standard
# (forward pass, calcul de perte, backward pass, optimisation des poids, évaluation).
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics
)

print("\n>> Demarrage du Fine-Tuning sur votre GTX 1650 (CUDA)...")
trainer.train()

# Sauvegarde finale du modèle fine-tuné et du tokenizer associé.
# Ce dossier `./fine_tuned_distilbert` sera ensuite utilisé par le moteur ABSA pour les prédictions.
print("\n>> Sauvegarde du modele entraine localement...")
model.save_pretrained("./fine_tuned_distilbert")
tokenizer.save_pretrained("./fine_tuned_distilbert")
print("[OK] Modele sauvegarde avec succes dans le dossier './fine_tuned_distilbert' !")