# 📱 Aspect-Based Sentiment Analysis (ABSA) pour le E-Commerce
## 🎯 Projet d'Étude Appliqué — Entreprise Fictive : **Electro-Marjane**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-blue?style=flat)](https://huggingface.co/docs/transformers/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)](https://sqlite.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat&logo=microsoft-power-bi&logoColor=black)](https://powerbi.microsoft.com/)

---

## 📖 1. Contexte Métier & Objectifs

### 📉 Problématique Métier
L'entreprise fictive **Electro-Marjane** a constaté une **baisse de 15 % de la fidélité client** sur la catégorie des accessoires de téléphones portables. Les avis clients bruts laissés sur la plateforme sont extrêmement volumineux et inexploitables à la main. 

Une analyse globale du sentiment (déterminer si un avis est simplement positif ou négatif) s'avère insuffisante, car elle masque les causes réelles d'insatisfaction. Par exemple :
> *« Le téléphone est super, l'écran est magnifique, mais la livraison a pris deux semaines et le colis est arrivé endommagé ! »*

Une approche classique classerait cet avis comme neutre ou mitigé. Or, pour Electro-Marjane, il y a deux signaux très distincts :
*   **Hardware / Produit** : Très Positif (Écran, design).
*   **Logistique / Service** : Très Négatif (Délai de livraison, état du colis).

### 🎯 Objectifs du Projet
Ce projet d'étude vise à concevoir un système complet d'**Analyse de Sentiment par Aspect (ABSA - Aspect-Based Sentiment Analysis)** permettant de :
1.  **Segmenter** les avis complexes en propositions sémantiques distinctes.
2.  **Associer** chaque segment à un aspect précis de l'offre commerciale (**Hardware / Design**, **Software / OS**, **Logistique / Packaging**, **Prix / Rapport qualité-prix**).
3.  **Prédire** avec précision le sentiment associé à chaque aspect (Positif, Neutre, Négatif).
4.  **Stocker et Visualiser** les résultats dans un tableau de bord décisionnel interactif pour orienter les actions correctives du SAV et de la direction.

---

## 🛠️ 2. Pile Technique (Tech Stack)

Le projet repose sur une architecture moderne de traitement de données et d'apprentissage profond :

| Composant | Technologie | Rôle / Description |
| :--- | :--- | :--- |
| **Langage & Frameworks** | **Python 3.10+**, **Pandas**, **Scikit-Learn** | Gestion des structures de données, manipulation et préparation des métriques d'évaluation. |
| **Calcul Profond (DL)** | **PyTorch (CUDA / GPU)** | Framework d'apprentissage pour l'entraînement et l'exécution du Transformer. |
| **NLP & Nettoyage** | **NLTK** | Tokenisation, nettoyage et filtrage des mots vides (*stopwords*). |
| **Modèle d'IA** | **Hugging Face Transformers** | Utilisation de l'architecture pré-entraînée **DistilBERT** (`distilbert-base-uncased`). |
| **Base de Données** | **SQLite 3** | Stockage relationnel local pour structurer et requêter les avis et leurs analyses. |
| **Visualisation** | **Microsoft Power BI Desktop** | Dashboard décisionnel connecté en direct à la base SQLite via un connecteur **ODBC**. |

---

## 📐 3. Architecture du Pipeline de Données (Data Flow)

Le flux de données se décompose en 5 phases majeures, de l'ingestion brute à la prise de décision :

```mermaid
flowchart TD
    subgraph Ingestion
        A["Fichier Source Amazon JSON<br>(141 Mo - Cell Phones & Accessories)"] --> B["Script de Fine-Tuning<br>(finetune_transformer.py)"]
        A --> C["Moteur d'Analyse NLP<br>(moteur_nlp_absa.py)"]
    end

    subgraph Apprentissage (Fine-Tuning)
        B -->|Entraînement GPU 1650 4Go + Mixed Precision| D["Modèle Fine-tuné localement<br>(./fine_tuned_distilbert)"]
    end

    subgraph Pipeline NLP & Inférence
        C -->|1. Nettoyage NLTK Stopwords| E["Texte Nettoyé"]
        E -->|2. Segmentation par Clauses (Regex)| F["Segments de Phrases"]
        F -->|3. Ontologie de Mots-Clés| G["Aspect Détecté"]
        G & D -->|4. Inférence DistilBERT Local| H["Sentiment Prédit par Aspect"]
    end

    subgraph Stockage Relationnel
        H -->|Insertion SQL| I[("Base SQLite<br>(electro_marjane.db)")]
        I -->|Table 1| I1["Avis_Bruts<br>(Métadonnées, Étoiles)"]
        I -->|Table 2| I2["Analyses_ABSA<br>(Aspects, Sentiments, Confiance)"]
    end

    subgraph Visualisation Décisionnelle
        I1 & I2 -->|Connexion ODBC| J["Tableau de bord Power BI<br>(dash.pbix)"]
        J --> K1["Rapport Décisionnel (Direction)"]
        J --> K2["Rapport Opérationnel (SAV / Alertes)"]
    end

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
    style J fill:#f96,stroke:#333,stroke-width:2px
```

### Détails des étapes du flux :
1.  **Ingestion & Préparation** : Le dataset d'origine contient des données massives d'avis Amazon (Cell Phones & Accessories). 
2.  **Fine-Tuning (Apprentissage)** : Pour adapter le modèle de langue générale DistilBERT à la sensibilité des avis de commerce (gestion fine du vocabulaire e-commerce), le script [finetune_transformer.py](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/finetune_transformer.py) ré-entraîne les couches supérieures du réseau sur GPU (Nvidia GTX 1650 4Go) en utilisant la précision mixte 16-bit (`fp16=True`) et des mini-lots de taille 4 pour ne pas saturer la mémoire vidéo (VRAM).
3.  **Pipeline d'Analyse ABSA** : Le script [moteur_nlp_absa.py](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/moteur_nlp_absa.py) découpe chaque avis sur les ponctuations et les conjonctions de coordination (`but`, `and`) afin d'isoler les propositions. Un dictionnaire d'aspects filtre les clauses par mots-clés, puis le modèle local classifie les segments concernés en 3 classes (Positif, Neutre, Négatif).
4.  **Stockage** : Les résultats sont injectés dans [electro_marjane.db](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/electro_marjane.db) sous deux tables normalisées pour éviter la redondance d'information.
5.  **Restitution** : Le fichier [dash.pbix](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/dash.pbix) offre deux perspectives de restitution : décisionnelle (pour suivre l'évolution de la fidélité globale) et opérationnelle (pour lister les alertes logistiques/SAV nécessitant des actions urgentes).

---

## 📂 4. Structure du Dossier du Projet

Voici l'organisation des fichiers au sein du dépôt :

*   📂 [**`fine_tuned_distilbert/`**](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/fine_tuned_distilbert) : Contient les poids du modèle d'IA fine-tuné localement, ses fichiers de configuration et le tokenizer.
*   📂 [**`results/`**](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/results) : Contient les points de contrôle (*checkpoints*) générés pendant l'entraînement du modèle.
*   📂 [**`logs/`**](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/logs) : Fichiers journaux d'entraînement contenant les pertes (*loss*) et l'accuracy pour le suivi (TensorBoard).
*   📂 [**`.vscode/`**](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/.vscode) : Fichiers de configuration spécifiques à l'éditeur de code Visual Studio Code.
*   📄 [**`Cell_Phones_and_Accessories_5.json`**](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/Cell_Phones_and_Accessories_5.json) : Le jeu de données d'origine (141 Mo) contenant les avis clients bruts au format JSON Lines.
*   💾 [**`electro_marjane.db`**](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/electro_marjane.db) : La base de données relationnelle SQLite 3 contenant les tables finales structurées.
*   📊 [**`dash.pbix`**](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/dash.pbix) : Le fichier source Microsoft Power BI Desktop pour l'affichage visuel et dynamique des KPIs.
*   📜 [**`finetune_transformer.py`**](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/finetune_transformer.py) : Script Python permettant d'adapter le modèle DistilBERT pré-entraîné à notre tâche de classification de sentiments.
*   📜 [**`moteur_nlp_absa.py`**](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/moteur_nlp_absa.py) : Script principal orchestrant la segmentation des avis, la détection des aspects et l'inférence par le modèle d'IA pour peupler la base SQLite.
*   📜 [**`evaluer_modele.py`**](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/evaluer_modele.py) : Script d'évaluation scientifique calculant les métriques d'apprentissage (Précision, Rappel, F1-Score, Matrice de Confusion).

---

## ⚙️ 5. Installation & Prérequis

### 📋 Prérequis système
*   **Python 3.10 ou version supérieure**
*   **Microsoft Power BI Desktop** (pour éditer ou visualiser le tableau de bord)
*   **Pilote SQLite ODBC** (ex: *SQLite ODBC Driver* de Christian Werner) pour lier Power BI à la base de données locale.
*   Une carte graphique NVIDIA compatible CUDA est fortement recommandée pour exécuter l'entraînement en moins de 15 minutes, mais le modèle peut fonctionner sur CPU pour l'inférence.

### 📥 Étape 1 : Cloner le dépôt et installer les dépendances
Ouvrez votre terminal et exécutez les commandes suivantes pour installer les packages requis :

```bash
# Installation des packages clés via pip
pip install pandas scikit-learn torch transformers tqdm
```

### 📥 Étape 2 : Configurer NLTK
Le moteur NLP utilise NLTK pour le filtrage linguistique. Les ressources nécessaires sont téléchargées automatiquement au premier lancement du script [moteur_nlp_absa.py](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/moteur_nlp_absa.py) via la commande intégrée :
```python
import nltk
nltk.download('stopwords')
```

---

## 🚀 6. Guide d'Utilisation

Pour reproduire l'intégralité de l'analyse, suivez rigoureusement l'ordre d'exécution suivant :

### Etape 1 : Ré-entraînement du modèle (Fine-tuning)
Lancez l'entraînement de DistilBERT sur le dataset brut. Cette étape va lire [Cell_Phones_and_Accessories_5.json](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/Cell_Phones_and_Accessories_5.json), tokeniser les avis et entraîner le réseau de neurones en sauvegardant le modèle final dans [fine_tuned_distilbert/](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/fine_tuned_distilbert).

```bash
python finetune_transformer.py
```
> 💡 *Note : L'entraînement utilise la précision mixte (FP16) pour accélérer le calcul sur GPU GTX 1650 4Go.*

### Etape 2 : Exécution du Moteur d'Analyse ABSA
Exécutez le pipeline principal. Il va segmenter les avis, détecter les aspects à l'aide de l'ontologie de mots-clés, appliquer le modèle d'IA fraîchement entraîné pour prédire le sentiment de chaque segment, et insérer le tout dans la base de données.

```bash
python moteur_nlp_absa.py
```
> 🚀 *À la fin de ce script, la base SQLite [electro_marjane.db](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/electro_marjane.db) est créée et intégralement alimentée.*

### Etape 3 : Évaluation du Modèle (Optionnel)
Vous pouvez mesurer scientifiquement la qualité des prédictions sur un ensemble de test indépendant en lançant :

```bash
python evaluer_modele.py
```

### Etape 4 : Visualisation sur Power BI
1.  Assurez-vous que le pilote **SQLite ODBC Driver** est correctement installé sur votre système Windows.
2.  Ouvrez le fichier [dash.pbix](file:///c:/Users/maniy/OneDrive/Desktop/Projet_NLTK_Tronsformers/dash.pbix) dans **Microsoft Power BI Desktop**.
3.  Cliquez sur le bouton **« Actualiser »** dans l'onglet Accueil pour charger les nouvelles données issues de la base SQLite locale.

---

## 📊 7. Résultats Scientifiques & Évaluation

Le modèle d'IA personnalisé a été rigoureusement validé sur un **jeu de test d'échappement strict** de **1 000 avis clients** (données totalement exclues de la phase d'entraînement). 

### 📈 Métriques Globales obtenues
Les performances obtenues mettent en évidence l'efficacité du fine-tuning de DistilBERT sur le corpus e-commerce :

| Métrique | Score | Interprétation |
| :--- | :--- | :--- |
| **Exactitude Globale (Accuracy)** | **85,60 %** | Plus de 8,5 prédictions sur 10 sont correctes toutes classes confondues. |
| **Précision (Classe Négative)** | **83,00 %** | Très faible taux de fausses alertes sur la détection des insatisfactions clients. |
| **F1-Score (Classe Neutre)** | **47,00 %** | Classe plus difficile à discerner en raison de l'ambiguïté naturelle des avis neutres. |

### 🔬 Analyse de la Généralisation
*   **Absence de sur-apprentissage (overfitting)** : L'écart minimal observé entre les performances d'entraînement et les données de test prouve que le modèle a appris des règles sémantiques robustes et réutilisables au lieu de simplement mémoriser les phrases d'entraînement.
*   **Pertinence pour le SAV** : La précision de 83 % sur la classe négative garantit que les alertes logistiques et de support générées dans Power BI pointent sur des réelles insatisfactions clients, optimisant ainsi le temps de traitement des équipes opérationnelles d'Electro-Marjane.

---
✍️ *Projet académique réalisé dans le cadre du module NLP & Deep Learning.*
