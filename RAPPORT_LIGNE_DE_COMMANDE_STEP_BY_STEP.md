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
8. [Étape 9 — Inférence en production (`src/predict.py`)](#étape-9--inférence-en-production-srcpredictpy)
9. [Réponses officielles aux Questions 1 à 19](#réponses-officielles-aux-questions-1-à-19)
10. [Validation des Mini-Défis 1 à 4](#validation-des-mini-défis-1-à-4)

---

## Étape 1 — Vérification de l'environnement Python

### Commande :
```powershell
python --version
```

### Sortie console obtenue :
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
> - `src/` : Contient les scripts du pipeline modulaire (`prepare.py`, `train.py`, `evaluate.py`, `predict.py`).
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

## Étape 9 — Inférence en production (`src/predict.py`)

### Commande :
```powershell
python src/predict.py
```

### Sortie console réelle :
```text
============================================================
[ETAPE] INFERENCE EN PRODUCTION (predict.py)
============================================================
Inference sur 3 nouveaux echantillons de production :

Echantillon 1 : [5.1, 3.5, 1.4, 0.2]
  -> Classe predite : SETOSA (ID: 0)
  -> Confiance      : 100.00%
  -> Probabilites   : {'setosa': 1.0, 'versicolor': 0.0, 'virginica': 0.0}

Echantillon 2 : [6.0, 2.9, 4.5, 1.5]
  -> Classe predite : VERSICOLOR (ID: 1)
  -> Confiance      : 99.00%
  -> Probabilites   : {'setosa': 0.0, 'versicolor': 0.99, 'virginica': 0.01}

Echantillon 3 : [6.9, 3.1, 5.4, 2.1]
  -> Classe predite : VIRGINICA (ID: 2)
  -> Confiance      : 100.00%
  -> Probabilites   : {'setosa': 0.0, 'versicolor': 0.0, 'virginica': 1.0}

============================================================
[OK] Inference terminee avec succes.
============================================================
```

---

## Monitoring et Data Drift (Questions 18 & 19)

### Situation analysée :
- Entraînement : Âge moyen = 35 ans.
- Production : Âge moyen = 58 ans.

> **Réponse Q18 : Y a-t-il potentiellement un Data Drift ?**  
> **Oui, absolument.** Il s'agit d'un **Covariate Shift** (dérive des variables explicatives $P(X)$). La moyenne est décalée de +23 ans, indiquant que la population en production est structurellement différente de celle sur laquelle le modèle a appris.

> **Réponse Q19 : Pourquoi cette évolution peut-elle affecter les prédictions ?**  
> Les frontières de décision et les seuils fixés dans les arbres du modèle ont été calibrés pour des profils de 35 ans. Sur des personnes de 58 ans, les relations entre variables changent et le modèle extrapole en dehors de son domaine de confiance. Cela entraîne une dégradation silencieuse des performances, nécessitant des outils de monitoring (alertes sur les distributions) et une boucle de réentraînement continu (*Continuous Training*).

---

## Synthèse finale des Mini-Défis

| Défi | Intitulé | Statut | Fichier associé |
| :--- | :--- | :---: | :--- |
| **Défi 1** | Créer `prepare.py`, `train.py`, `evaluate.py`, `predict.py` | ✅ Réalisé | `src/` |
| **Défi 2** | Créer la fonction réutilisable `train_model()` | ✅ Réalisé | `src/train.py` |
| **Défi 3** | Logger automatiquement Accuracy, Precision, Recall, F1 dans MLflow | ✅ Réalisé | `src/train.py` + `mlruns/` |
| **Défi 4** | Tester au moins 3 configurations (50, 100, 200) et comparer | ✅ Réalisé | `experiments/benchmark_results.csv` |
