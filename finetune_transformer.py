import pandas as pd
import torch
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Configuration
NOM_MODELE = "distilbert-base-uncased"  
TAILLE_ECHANTILLON = 30000            
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Périphérique d'entraînement : {device.upper()}")

# Chargement du dataset brut
print("Chargement des données...")
if TAILLE_ECHANTILLON:
    df = pd.read_json("Cell_Phones_and_Accessories_5.json", lines=True).head(TAILLE_ECHANTILLON)
else:
    df = pd.read_json("Cell_Phones_and_Accessories_5.json", lines=True)

df['Texte'] = df['summary'].fillna('') + ". " + df['reviewText'].fillna('')

# Mappage des notes (étoiles) en 3 classes (0: Négatif, 1: Neutre, 2: Positif)
def convertir_etoiles_en_label(stars):
    if stars >= 4:
        return 2
    elif stars <= 2:
        return 0
    else:
        return 1

df['label'] = df['overall'].apply(convertir_etoiles_en_label)

# Division Train (80%) / Validation (20%)
textes_train, textes_val, labels_train, labels_val = train_test_split(
    df['Texte'].tolist(), 
    df['label'].tolist(), 
    test_size=0.2,       
    random_state=42      
)

# Tokenisation (DistilBERT)
print("Initialisation du Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(NOM_MODELE)

print("Tokenisation des textes...")
train_encodings = tokenizer(textes_train, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(textes_val, truncation=True, padding=True, max_length=128)

# Dataset PyTorch pour Hugging Face Trainer
class AmazonReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset_train = AmazonReviewsDataset(train_encodings, labels_train)
dataset_val = AmazonReviewsDataset(val_encodings, labels_val)

# Chargement et hyperparamètres du modèle (optimisés pour VRAM 4Go)
print("Chargement du modèle de classification...")
model = AutoModelForSequenceClassification.from_pretrained(NOM_MODELE, num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",           
    num_train_epochs=2,              
    per_device_train_batch_size=4,   # Réduit pour éviter la saturation VRAM 4Go (ex: GTX 1650)
    per_device_eval_batch_size=4,    
    warmup_steps=100,                
    weight_decay=0.01,               
    logging_dir="./logs",            
    logging_steps=50,                
    eval_strategy="epoch",           
    save_strategy="epoch",           
    fp16=True,                       # Précision mixte 16-bit (indispensable sur GPU 4Go)
    load_best_model_at_end=True,     
    metric_for_best_model="accuracy" 
)

# Calcul de l'Exactitude globale (Accuracy)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Initialisation du Trainer et Entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics
)

print("\n>> Démarrage du Fine-Tuning...")
trainer.train()

# Sauvegarde locale du modèle et du tokenizer associés
print("\n>> Sauvegarde du modèle affiné...")
model.save_pretrained("./fine_tuned_distilbert")
tokenizer.save_pretrained("./fine_tuned_distilbert")
print("[OK] Sauvegarde réussie dans './fine_tuned_distilbert'")