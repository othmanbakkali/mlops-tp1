# RAPPORT D'EXÉCUTION LIGNE DE COMMANDE — STEP BY STEP
## TP 1 : Mise en œuvre des fondamentaux MLOps (Chapitre 1)

> **Étudiant :** BAKKALI YEDRI Othman  
> **Établissement :** EMSI  
> **Système d'exploitation :** Windows (PowerShell)  
> **Version Python :** Python 3.12.6  

---

## 📋 Table des Matières

1. [Étape 1 — Vérification de l'environnement Python](#étape-1--vérification-de-lenvironnement-python)
2. [Étape 2 — Création de l'arborescence du projet](#étape-2--création-de-larborescence-du-projet)
3. [Étape 3 & 4 — Environnement virtuel et dépendances](#étape-3--4--environnement-virtuel-et-dépendances)
4. [Étape 5 — Préparation et découpage des données (`src/prepare.py`)](#étape-5--préparation-et-découpage-des-données-srcpreparepy)
5. [Étape 6 — Entraînement et suivi MLflow (`src/train.py`)](#étape-6--entraînement-et-suivi-mlflow-srctrainpy)
6. [Étape 7 — Visualisation de l'interface MLflow](#étape-7--visualisation-de-linterface-mlflow)
7. [Étape 8 — Évaluation du modèle (`src/evaluate.py`)](#étape-8--évaluation-du-modèle-srcevaluatepy)
8. [Étape 9 — Charger le modèle et prédire (`src/predict.py`)](#étape-9--charger-le-modèle-et-prédire-srcpredictpy)
9. [Étape 10 — Première activité de Data Drift (`src/drift.py`)](#étape-10--première-activité-de-data-drift-srcdriftpy)
10. [Étape 11 — Versionnement avec Git](#étape-11--versionnement-avec-git)
11. [Étape 12 — Organisation finale du projet](#étape-12--organisation-finale-du-projet)
12. [Étape 13 — Livrables remis](#étape-13--livrables-remis)
13. [Étape 14 — Chaîne MLOps complète réalisée](#étape-14--chaîne-mlops-complète-réalisée)
14. [Réponses consolidées aux Questions 1 à 19](#réponses-consolidées-aux-questions-1-à-19)

---

## Étape 1 — Vérification de l'environnement Python

### Commande :
```powershell
python --version
```

### Sortie console réelle :
```text
Python 3.12.6
```

> **Réponse Q1 :** La version utilisée est **Python 3.12.6** (Windows 64 bits).

---

## Étape 2 — Création de l'arborescence du projet

### Commandes :
```powershell
mkdir mlops-tp1
cd mlops-tp1
mkdir data\raw, data\processed, models, src, experiments
```

### Vérification de l'arborescence :
```powershell
tree /F
```

### Résultat de l'arborescence :
```text
mlops-tp1/
│
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
├── experiments/
├── requirements.txt
└── README.md
```

> **Réponse Q2 — Rôle de chaque dossier :**
> - `data/` : Centralise les jeux de données brutes (`raw/`) et transformées (`processed/`).
> - `models/` : Héberge les modèles sérialisés (`iris_model.pkl`) prêts au déploiement.
> - `src/` : Contient les scripts du pipeline modulaire (`prepare.py`, `train.py`, `evaluate.py`, `predict.py`, `drift.py`).
> - `experiments/` : Archive les résumés de runs, rapports et graphiques.
> - `requirements.txt` : Fige la liste des dépendances pour garantir la reproductibilité.
> - `README.md` : Documente le projet, les métriques et les réponses du TP.

---

## Étape 3 & 4 — Environnement virtuel et dépendances

### Commandes :
```powershell
# Création de l'environnement virtuel
python -m venv .venv

# Activation de l'environnement virtuel sous Windows
.venv\Scripts\activate

# Installation des bibliothèques MLOps requises
pip install pandas numpy scikit-learn matplotlib mlflow joblib

# Export des dépendances figées
pip freeze > requirements.txt
```

> **Réponse Q3 :** Conserver `requirements.txt` est indispensable pour assurer la **reproductibilité de l'environnement**. Cela permet à n'importe quel collaborateur ou serveur d'intégration continue de recréer à l'identique l'environnement logiciel sans conflit de version.

---

## Étape 5 — Préparation et découpage des données (`src/prepare.py`)

Ce script charge le jeu Iris, affiche ses propriétés, isole les données brutes dans `data/raw/iris_raw.csv` et génère un découpage stratifié 80% train / 20% test dans `data/processed/`.

### Commande :
```powershell
python src/prepare.py
```

### Sortie console réelle :
```text
============================================================
[ETAPE] PREPARATION DES DONNEES (prepare.py)
============================================================
Dataset charge avec succes :
  - Nombre total d'observations : 150
  - Nombre de variables d'entree : 4
  - Variables : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
  - Classes cibles : ['setosa', 'versicolor', 'virginica'] (labels: [0, 1, 2])

[OK] Donnees brutes sauvegardees dans : data\raw\iris_raw.csv

Separation Train/Test effectuee (test_size=0.2, random_state=42) :
  - Train set : 120 exemples (4 features)
  - Test set  : 30 exemples (4 features)
[OK] Train dataset sauvegarde dans : data\processed\train.csv
[OK] Test dataset sauvegarde dans  : data\processed\test.csv
============================================================
```

> **Réponses aux questions :**
> - **Q4 :** Le dataset contient **150 observations**.
> - **Q5 :** Il possède **4 variables d'entrée**.
> - **Q6 :** On sépare les données en Train/Test pour évaluer la capacité de généralisation du modèle et détecter le sur-apprentissage (*overfitting*).
> - **Q7 :** `random_state=42` fixe la graine du générateur aléatoire pour garantir un découpage déterministe et reproductible.

---

## Étape 6 — Entraînement et suivi MLflow (`src/train.py`)

Le script `src/train.py` implémente la fonction `train_model()` (Défi 2), logge automatiquement dans MLflow les paramètres, l'Accuracy, la Precision, le Recall et le F1-score (Défi 3) et exécute le benchmark multi-configurations (Défi 4 : 50, 100 et 200 arbres).

### Commande :
```powershell
python src/train.py --benchmark
```

### Sortie console réelle :
```text
============================================================
[DEFI 4] BENCHMARK MULTI-CONFIGURATIONS (50, 100, 200)
============================================================

[INFO] Lancement de l'experience : n_estimators=50, random_state=42
  Resultats du Run :
    - Accuracy  : 0.9000
    - Precision : 0.9024
    - Recall    : 0.9000
    - F1-score  : 0.8997
  [OK] Modele serialise sauvegarde dans : models/iris_model.pkl

[INFO] Lancement de l'experience : n_estimators=100, random_state=42
  Resultats du Run :
    - Accuracy  : 0.9000
    - Precision : 0.9024
    - Recall    : 0.9000
    - F1-score  : 0.8997
  [OK] Modele serialise sauvegarde dans : models/iris_model.pkl

[INFO] Lancement de l'experience : n_estimators=200, random_state=42
  Resultats du Run :
    - Accuracy  : 0.9000
    - Precision : 0.9024
    - Recall    : 0.9000
    - F1-score  : 0.8997
  [OK] Modele serialise sauvegarde dans : models/iris_model.pkl

============================================================
[RESULTAT] TABLEAU COMPARATIF DES EXPERIENCES (MLflow)
============================================================
                Run  n_estimators Accuracy Precision Recall F1-score
 RF_n_estimators_50            50   0.9000    0.9024 0.9000   0.8997
RF_n_estimators_100           100   0.9000    0.9024 0.9000   0.8997
RF_n_estimators_200           200   0.9000    0.9024 0.9000   0.8997

[OK] Resultats sauvegardes dans : experiments/benchmark_results.csv
============================================================
```

> **Réponse Q10 :** On sauvegarde le modèle dans `models/iris_model.pkl` pour pouvoir l'utiliser en production lors des phases d'inférence sans devoir le réentraîner à chaque appel.

---

## Étape 7 — Visualisation de l'interface MLflow

### Commande pour démarrer l'UI :
```powershell
mlflow ui
```

Accéder à l'interface via le navigateur web :
👉 **`http://127.0.0.1:5000`**

### Éléments enregistrés et consultables dans l'UI :
- **Expérience :** `TP1_Iris_MLOps`
- **Runs :** `RF_n_estimators_50`, `RF_n_estimators_100`, `RF_n_estimators_200`
- **Paramètres :** `n_estimators`, `random_state`
- **Métriques :** `accuracy`, `precision`, `recall`, `f1_score`
- **Artefacts :** Modèle Scikit-learn loggé avec son fichier `MLmodel` et ses dépendances.

> **Réponses aux questions :**
> - **Q11 :** MLflow assure le suivi d'expériences (*Experiment Tracking*), le versionnement des modèles (*Model Registry*) et la traçabilité complète des métadonnées ML.
> - **Q12 :** Paramètres enregistrés : `n_estimators` et `random_state`.
> - **Q13 :** Métriques enregistrées : `accuracy`, `precision`, `recall`, `f1_score`.
> - **Q14 :** Cette traçabilité permet d'auditer chaque version, de comprendre l'effet des hyperparamètres, et de garantir la conformité et la reproductibilité du cycle MLOps.
> - **Q15 :** Les 3 expériences obtiennent une Accuracy équivalente de **0.9000 (90.00%)**.
> - **Q16 :** Seul l'hyperparamètre `n_estimators` (nombre d'arbres) a varié.
> - **Q17 :** Conserver ces informations permet de choisir la solution la plus économique en ressources : 50 arbres offrent la même performance que 200, réduisant la latence et l'empreinte mémoire.

---

## Étape 8 — Évaluation du modèle (`src/evaluate.py`)

### Commande :
```powershell
python src/evaluate.py
```

### Sortie console réelle :
```text
============================================================
[ETAPE] EVALUATION DU MODELE (evaluate.py)
============================================================
Modele evalue : models/iris_model.pkl
Jeu de test   : 30 observations

[METRIQUE] ACCURACY : 0.9000 (90.00%)

[RAPPORT DE CLASSIFICATION] :
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.82      0.90      0.86        10
   virginica       0.89      0.80      0.84        10

    accuracy                           0.90        30
   macro avg       0.90      0.90      0.90        30
weighted avg       0.90      0.90      0.90        30

Matrice de confusion :
[[10  0  0]
 [ 0  9  1]
 [ 0  2  8]]

[OK] Rapport texte sauvegarde dans : experiments\evaluation_report.txt
[OK] Matrice de confusion sauvegardee dans : experiments\confusion_matrix.png
============================================================
```

> **Réponses aux questions :**
> - **Q8 :** L'Accuracy obtenue est **0.9000 (90.00%)**.
> - **Q9 :**
>   - **Precision :** Exactitude des prédictions positives ($\frac{TP}{TP + FP}$).
>   - **Recall :** Capacité à trouver tous les vrais positifs ($\frac{TP}{TP + FN}$).
>   - **F1-score :** Moyenne harmonique entre Precision et Recall ($2 \times \frac{P \times R}{P + R}$).

---

## Étape 9 — Charger le modèle et prédire (`src/predict.py`)

### Code de `src/predict.py` :
```python
import joblib
from sklearn.datasets import load_iris

model = joblib.load("models/iris_model.pkl")
iris = load_iris()
sample = [iris.data[0]]
prediction = model.predict(sample)

print("Classe :", prediction[0])
print("Nom :", iris.target_names[prediction[0]])
```

### Commande :
```powershell
python src/predict.py
```

### Sortie console réelle :
```text
Classe : 0
Nom : setosa
```

---

## Étape 10 — Première activité de Data Drift (`src/drift.py`)

### Principe :
Le **Data Drift** correspond à une modification de la distribution statistique des données d'entrée entre l'entraînement et la production ($P_{prod}(X) \neq P_{train}(X)$).

### Code d'observation des statistiques :
```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.describe())
```

### Commande :
```powershell
python src/drift.py
```

### Sortie console réelle :
```text
=================================================================
[ACTIVITÉ DATA DRIFT] STATISTIQUES DESCRIPTIVES DU DATASET INITIAL
=================================================================
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
count         150.000000        150.000000         150.000000        150.000000
mean            5.843333          3.057333           3.758000          1.199333
std             0.828066          0.435866           1.765298          0.762238
min             4.300000          2.000000           1.000000          0.100000
25%             5.100000          2.800000           1.600000          0.300000
50%             5.800000          3.000000           4.350000          1.300000
75%             6.400000          3.300000           5.100000          1.800000
max             7.900000          4.400000           6.900000          2.500000

=================================================================
```

### Question :
**Quelles variables pourraient changer en production ? Quel impact ce changement pourrait-il avoir sur les prédictions ?**

> **Réponse détaillée :**
> 1. **Variables susceptibles de changer :**
>    - Les longueurs et largeurs de pétales (`petal length`, `petal width`) et de sépales (`sepal length`, `sepal width`) peuvent dériver si les fleurs cueillies en production proviennent d'une autre région géographique, d'un climat différent (sécheresse ou humidité modifiant la taille) ou d'une période de floraison différente.
>    - Les mesures peuvent également dériver suite à un changement de capteur ou de méthode de mesure (biais d'instrumentation).
> 2. **Impact sur les prédictions :**
>    - Les arbres de décision du Random Forest s'appuient sur des seuils stricts appris sur la distribution d'origine (ex: `petal length <= 2.45` pour Setosa).
>    - Si la distribution glisse (ex: une augmentation moyenne de la taille), le modèle va prédire avec une confiance faussée, confondre des classes (ex: Versicolor classée Virginica), ou générer une dégradation silencieuse sans qu'aucune erreur logicielle ne se déclenche.
>    - Cela illustre pourquoi le **monitoring continu des distributions** (tests KS, PSI) et une boucle de réentraînement automatique (*Continuous Training*) sont indispensables en MLOps.

---

## Étape 11 — Versionnement avec Git

### 1. Initialiser le dépôt
```powershell
git init
```
**Sortie réelle :**
```text
Initialized empty Git repository in C:/Users/othma/OneDrive/Bureau/EMSI/MLOPS/TP/Chapitre 1/mlops-tp1/.git/
```

### 2. Créer `.gitignore`
Contenu du fichier `.gitignore` :
```text
.venv/
__pycache__/
*.pyc
mlruns/
models/*.pkl
```

### 3. Vérifier le statut
```powershell
git status
```
**Sortie réelle :**
```text
On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.gitignore
	RAPPORT_LIGNE_DE_COMMANDE_STEP_BY_STEP.md
	README.md
	data/
	experiments/
	requirements.txt
	src/
```

### 4. Premier commit
```powershell
git add .
git commit -m "TP1 MLOps - premier modèle Iris"
```
**Sortie réelle :**
```text
[master (root-commit) e276594] TP1 MLOps - premier modèle Iris
 15 files changed, 1312 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 RAPPORT_LIGNE_DE_COMMANDE_STEP_BY_STEP.md
 create mode 100644 README.md
 create mode 100644 data/processed/test.csv
 create mode 100644 data/processed/train.csv
 create mode 100644 data/raw/iris_raw.csv
 create mode 100644 experiments/benchmark_results.csv
 create mode 100644 experiments/confusion_matrix.png
 create mode 100644 experiments/evaluation_report.txt
 create mode 100644 requirements.txt
 create mode 100644 src/drift.py
 create mode 100644 src/evaluate.py
 create mode 100644 src/predict.py
 create mode 100644 src/prepare.py
 create mode 100644 src/train.py
```

### 5. Vérifier l'historique
```powershell
git log --oneline
```
**Sortie réelle :**
```text
e276594 TP1 MLOps - premier modèle Iris
```

---

## Étape 12 — Organisation finale du projet

```text
mlops-tp1/
│
├── .venv/                         <- Environnement virtuel Python isolé
├── data/
│   ├── raw/
│   │   └── iris_raw.csv           <- Données brutes exportées
│   └── processed/
│       ├── train.csv              <- Données d'entraînement (120 lignes)
│       └── test.csv               <- Données de test (30 lignes)
│
├── experiments/
│   ├── benchmark_results.csv      <- Résultats des runs MLflow (50, 100, 200)
│   ├── evaluation_report.txt      <- Rapport de classification textuel
│   └── confusion_matrix.png       <- Matrice de confusion générée
│
├── models/
│   └── iris_model.pkl             <- Modèle sérialisé prêt pour l'inférence
│
├── src/
│   ├── prepare.py                 <- Découpage stratifié et préparation
│   ├── train.py                   <- Entraînement et tracking MLflow
│   ├── evaluate.py                <- Rapport de classification complet
│   ├── predict.py                 <- Script d'inférence (conforme atelier)
│   └── drift.py                   <- Analyse des statistiques de base et Data Drift
│
├── mlruns/                        <- Base locale d'expériences MLflow
├── .gitignore                     <- Exclusion des binaires, modèles et venv
├── requirements.txt               <- Dépendances figées du projet
├── README.md                      <- Rapport technique et académique
└── RAPPORT_LIGNE_DE_COMMANDE_STEP_BY_STEP.md <- Ce rapport d'exécution
```

---

## Étape 13 — Livrables remis

Le dossier `mlops-tp1/` contient :
1. Les scripts Python modulaires : `prepare.py`, `train.py`, `evaluate.py`, `predict.py`, `drift.py`.
2. Le fichier de dépendances figées : `requirements.txt`.
3. Le fichier d'exclusion Git : `.gitignore`.
4. Le modèle généré et sérialisé : `models/iris_model.pkl`.
5. Les résultats des expériences MLflow : répertoire `mlruns/` et `experiments/benchmark_results.csv`.
6. La matrice de confusion graphique : `experiments/confusion_matrix.png`.
7. Le rapport complet et l'analyse de Data Drift dans `README.md` et `RAPPORT_LIGNE_DE_COMMANDE_STEP_BY_STEP.md`.

---

## Étape 14 — Chaîne MLOps complète réalisée

```text
Windows (OS)
   ↓
Environnement virtuel (.venv)
   ↓
Données (Iris -> data/raw/ & data/processed/)
   ↓
Entraînement (Random Forest avec train_model())
   ↓
Évaluation (Accuracy, Precision, Recall, F1-score, Matrice de confusion)
   ↓
Sauvegarde (models/iris_model.pkl)
   ↓
MLflow (Tracking des runs, paramètres, métriques, artefacts)
   ↓
Comparaison (Benchmark 50 vs 100 vs 200 arbres)
   ↓
Prédiction (src/predict.py -> Classe: 0, Nom: setosa)
   ↓
Data Drift (src/drift.py -> Analyse de distribution et décalages)
   ↓
Git (Versionnement, .gitignore et commit initial)
```

> **Principe illustré :**  
> **Automatiser, expérimenter, versionner, mesurer et surveiller un modèle de Machine Learning** de bout en bout selon les fondamentaux MLOps.
