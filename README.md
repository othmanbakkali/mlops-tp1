# TP 1 — Mise en œuvre des fondamentaux MLOps : De la donnée au modèle reproductible

> **Auteur :** BAKKALI YEDRI Othman  
> **Module :** MLOps — Chapitre 1 : Fondamentaux des MLOps  
> **École :** EMSI  
> **Outils :** Python 3.12, Scikit-learn, MLflow, Pandas, Joblib  

---

## 1. Objectif du projet

Ce projet a pour objectif d'implémenter l'ensemble du cycle de vie MLOps à travers un cas pratique complet :
- Structurer un projet Machine Learning selon les bonnes pratiques industrielles et les standards MLOps.
- Mettre en place un environnement reproductible (code, dépendances, données, modèle et paramètres).
- Préparer et découper un jeu de données de manière stratifiée et déterministe.
- Modulariser le code source (`prepare.py`, `train.py`, `evaluate.py`, `predict.py`).
- Suivre les métriques d'évaluation et versionner les artefacts du modèle grâce à **MLflow Tracking**.
- Comparer différentes configurations d'hyperparamètres et analyser la reproductibilité.
- Comprendre et simuler les enjeux de **Monitoring** et de **Data Drift** lors du passage en production.

---

## 2. Dataset utilisé

Le dataset exploité est le célèbre jeu de données **Iris** issu de Fisher (1936), intégré à Scikit-learn :
- **Observations totales :** 150 échantillons.
- **Variables d'entrée (features, $X$) :** 4 caractéristiques morphologiques continues en centimètres :
  1. `sepal length (cm)` (longueur du sépale)
  2. `sepal width (cm)` (largeur du sépale)
  3. `petal length (cm)` (longueur du pétale)
  4. `petal width (cm)` (largeur du pétale)
- **Variable cible ($y$) :** Espèce d'iris (3 classes équilibrées de 50 individus chacune) :
  - `0` : Iris Setosa
  - `1` : Iris Versicolor
  - `2` : Iris Virginica
- **Partitionnement :**
  - **Train set (80%) :** 120 échantillons (stratifié, 40 par classe).
  - **Test set (20%) :** 30 échantillons (stratifié, 10 par classe).

---

## 3. Algorithme choisi

L'algorithme utilisé est un **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`).
- **Principe :** Méthode d'ensemble par *bagging* (Bootstrap Aggregating) combinant une multitude d'arbres de décision indépendants entraînés sur des sous-échantillons aléatoires avec remplacement. La décision finale découle d'un vote majoritaire des arbres.
- **Avantages MLOps :**
  - Robustesse face au sur-apprentissage (*overfitting*).
  - Gestion naturelle des relations non linéaires.
  - Hyperparamètre clé facilement modifiable (`n_estimators`) pour illustrer le suivi d'expériences sous MLflow.

---

## 4. Paramètres et Hyperparamètres

- **`random_state = 42`** : Graine aléatoire fixée pour le partitionnement `train_test_split` et le modèle `RandomForestClassifier`, garantissant une **reproductibilité déterministe stricte**.
- **`test_size = 0.2`** : 20% des données réservées au test.
- **`stratify = y`** : Préservation rigoureuse des proportions de chaque classe dans le train et le test.
- **`n_estimators`** : Nombre d'arbres de décision testés :
  - Configuration 1 : `50`
  - Configuration 2 : `100` (baseline)
  - Configuration 3 : `200`

---

## 5. Métriques obtenues

Évaluation détaillée sur le jeu de test (30 échantillons, 10 par classe) pour `n_estimators = 100` :

### Métriques globales
- **Accuracy :** **0.9000 (90.00%)**
- **Macro Average F1 :** **0.9000**
- **Weighted Average F1 :** **0.8997**

### Rapport de classification par classe
| Classe | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Setosa** | 1.00 | 1.00 | 1.00 | 10 |
| **Versicolor** | 0.82 | 0.90 | 0.86 | 10 |
| **Virginica** | 0.89 | 0.80 | 0.84 | 10 |

### Matrice de Confusion
```text
                  Prédit Setosa   Prédit Versicolor   Prédit Virginica
Réel Setosa            10                 0                   0
Réel Versicolor         0                 9                   1
Réel Virginica          0                 2                   8
```
- **Setosa :** 100% de détection sans aucune confusion (séparabilité linéaire parfaite).
- **Versicolor / Virginica :** Légère ambiguïté morphologique sur 3 fleurs proches de la frontière de décision (1 Versicolor classée Virginica, 2 Virginica classées Versicolor).

---

## 6. Résultats des expériences (Benchmark MLflow)

Suivi automatique sous l'expérience MLflow `TP1_Iris_MLOps` :

| Run | n_estimators | random_state | Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) | Modèle loggé |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1** | 50 | 42 | 0.9000 | 0.9024 | 0.9000 | 0.8997 | `mlflow.sklearn` |
| **Run 2** | 100 | 42 | 0.9000 | 0.9024 | 0.9000 | 0.8997 | `mlflow.sklearn` |
| **Run 3** | 200 | 42 | 0.9000 | 0.9024 | 0.9000 | 0.8997 | `mlflow.sklearn` |

### Analyse comparative
- Sur ce dataset de petite dimension (150 lignes), augmenter `n_estimators` au-delà de 50 arbres stabilise la variance mais ne modifie pas l'Accuracy globale sur les 30 exemples de test.
- Grâce à MLflow, chaque exécution est horodatée, avec les métadonnées Git, les paramètres exacts, les métriques et les artefacts du modèle sérialisé.

---

## 7. Structure du projet

```text
mlops-tp1/
│
├── data/
│   ├── raw/
│   │   └── iris_raw.csv           <- Données brutes exportées
│   └── processed/
│       ├── train.csv              <- Données d'entraînement (120 lignes)
│       └── test.csv               <- Données de test (30 lignes)
│
├── models/
│   └── iris_model.pkl             <- Modèle sérialisé prêt pour l'inférence
│
├── src/
│   ├── prepare.py                 <- Préparation, découpage stratifié et persistance
│   ├── train.py                   <- Entraînement avec train_model() et tracking MLflow
│   ├── evaluate.py                <- Rapport complet et matrice de confusion
│   └── predict.py                 <- Inférence unitaire et par lots en production
│
├── experiments/
│   ├── benchmark_results.csv      <- Comparatif des runs (50, 100, 200)
│   ├── evaluation_report.txt      <- Métriques détaillées au format texte
│   └── confusion_matrix.png       <- Matrice de confusion graphique
│
├── mlruns/                        <- Base de tracking locale MLflow
├── requirements.txt               <- Dépendances figées du projet
└── README.md                      <- Documentation complète
```

---

## 8. Réponses aux Questions du TP (Q1 à Q19)

### Partie 1 — Mise en place de l'environnement
- **Question 1 : Quelle version de Python utilisez-vous ?**  
  > **Réponse :** `Python 3.12.6` (64-bit sur architecture Windows).

- **Question 2 : Quel est le rôle de chacun des dossiers ?**  
  > - `data/` : Centralise les jeux de données (`raw/` pour les données brutes immutables, `processed/` pour les données nettoyées et découpées).  
  > - `models/` : Stocke les modèles entraînés et sérialisés (`.pkl`, `.onnx`, etc.) prêts à être déployés.  
  > - `src/` : Contient le code source modulaire du pipeline (ETL, entraînement, évaluation, inférence).  
  > - `experiments/` : Archive les rapports, visualisations et résultats comparatifs des différentes itérations.  
  > - `requirements.txt` : Liste figée des dépendances et de leurs versions pour reproduire l'environnement.  
  > - `README.md` : Guide d'utilisation, documentation technique et synthèse des résultats.

### Partie 2 — Environnement reproductible
- **Question 3 : Pourquoi est-il important de conserver `requirements.txt` ?**  
  > **Réponse :** Pour garantir la **reproductibilité de l'environnement d'exécution**. Sans fichier de dépendances versionné, les variations de versions de bibliothèques (ex: Scikit-learn, Numpy, Pandas) peuvent introduire des incompatibilités d'API, des modifications d'implémentation algorithmique ou des résultats différents lors du réentraînement ou du déploiement.

### Partie 3 — Préparation des données
- **Question 4 : Combien d'observations contient le dataset ?**  
  > **Réponse :** **150 observations** (50 pour chaque espèce d'Iris).

- **Question 5 : Combien de variables d'entrée possède-t-il ?**  
  > **Réponse :** **4 variables d'entrée numériques continues** : `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)` et `petal width (cm)`.

### Partie 4 — Séparation Train / Test
- **Question 6 : Pourquoi sépare-t-on les données en données d'entraînement et données de test ?**  
  > **Réponse :** Pour évaluer la capacité de **généralisation** du modèle sur des données nouvelles qu'il n'a jamais vues, et pour détecter l'**overfitting** (sur-apprentissage, lorsque le modèle mémorise le bruit du train set sans apprendre les vrais motifs sous-jacents).

- **Question 7 : À quoi sert `random_state=42` ?**  
  > **Réponse :** `random_state=42` initialise le générateur de nombres pseudo-aléatoires à une valeur fixe. Cela garantit que chaque exécution du code produira exactement le même découpage des données et les mêmes tirages d'arbres, assurant ainsi la **reproductibilité scientifique et technique**.

### Partie 6 — Évaluation
- **Question 8 : Quelle est l'Accuracy obtenue ?**  
  > **Réponse :** **0.9000 (soit 90.00%)** sur le jeu de test de 30 échantillons.

- **Question 9 : Que représentent Precision, Recall et F1-score ?**  
  > - **Precision ($\frac{TP}{TP+FP}$) :** La proportion de prédictions positives qui sont réellement correctes (fiabilité de la prédiction positive).  
  > - **Recall ($\frac{TP}{TP+FN}$) :** La proportion d'exemples réels positifs détectés par le modèle (exhaustivité/sensibilité).  
  > - **F1-score ($2 \times \frac{Precision \times Recall}{Precision + Recall}$) :** La moyenne harmonique de la précision et du rappel, idéale en cas de classes déséquilibrées ou de compromis précision/rappel.

### Partie 7 — Sauvegarde du modèle
- **Question 10 : Pourquoi doit-on sauvegarder le modèle ?**  
  > **Réponse :** Pour découpler la phase d'entraînement (souvent coûteuse en temps et calcul) de la phase d'**inférence en production**. Le modèle sérialisé (`iris_model.pkl`) peut être chargé instantanément par une API REST (FastAPI, Flask) ou un microservice pour répondre aux requêtes temps-réel sans réentraînement.

### Partie 9 — MLflow Tracking
- **Question 11 : Quel est le rôle de MLflow ?**  
  > **Réponse :** MLflow est une plateforme open-source de gestion du cycle de vie ML. Son composant **MLflow Tracking** permet d'enregistrer, d'auditer, de comparer et de visualiser les expériences (hyperparamètres, métriques de performance, artefacts de modèles et dépendances associées).

- **Question 12 : Quels paramètres avez-vous enregistrés ?**  
  > **Réponse :** `n_estimators` (nombre d'arbres : 50, 100, 200) et `random_state` (42).

- **Question 13 : Quelle métrique avez-vous enregistrée ?**  
  > **Réponse :** `accuracy`, complétée dans le mini-défi par `precision`, `recall` et `f1_score`.

- **Question 14 : Pourquoi cette traçabilité est-elle importante en MLOps ?**  
  > **Réponse :** La traçabilité permet :
  > 1. D'auditer l'historique de chaque version de modèle.
  > 2. De corréler directement un changement d'hyperparamètre ou de code à un gain/perte de performance.
  > 3. De sélectionner objectivement le meilleur modèle avant mise en production.
  > 4. De satisfaire aux exigences de gouvernance, de conformité et de reproductibilité d'entreprise.

### Partie 10 — Reproductibilité et comparaison
- **Question 15 : Quelle expérience obtient la meilleure Accuracy ?**  
  > **Réponse :** Les trois expériences (50, 100 et 200 arbres) obtiennent une Accuracy identique de **0.9000 (90%)**.

- **Question 16 : Qu'est-ce qui a changé entre les deux expériences ?**  
  > **Réponse :** La valeur de l'hyperparamètre `n_estimators` (la taille de la forêt aléatoire).

- **Question 17 : Pourquoi est-il important de conserver ces informations ?**  
  > **Réponse :** Pour éviter le travail à l'aveugle, pour documenter les choix techniques et pour optimiser le compromis performance/complexité. Par exemple, si 50 arbres donnent la même performance que 200 arbres, choisir le modèle à 50 arbres permet de diviser la taille mémoire et le temps d'inférence par 4.

### Partie 11 — Monitoring et Data Drift
- **Question 18 : Y a-t-il potentiellement un Data Drift si l'âge moyen passe de 35 ans à 58 ans ?**  
  > **Réponse :** **OUI, il y a un Data Drift majeur (Covariate Shift).** La distribution statistique de la variable d'entrée en production ($P_{prod}(X)$) est significativement décalée par rapport à celle vue lors de l'entraînement ($P_{train}(X)$).

- **Question 19 : Pourquoi cette évolution peut-elle affecter les prédictions ?**  
  > **Réponse :** Le modèle a été optimisé sur des motifs statistiques propres à une population d'âge moyen 35 ans. Face à une population de 58 ans, les règles de décision et les seuils appris dans les arbres ne sont plus représentatifs. Cela engendre une dégradation silencieuse des performances (chute de précision, biais de prédiction), justifiant la mise en place impérative d'un **monitoring continu** des données entrantes et du réentraînement automatique (*Continuous Training*).

---

## 9. Difficultés rencontrées et solutions apportées

1. **Encodage de la console sous Windows (cp1252 vs UTF-8) :**
   - *Problème :* Les caractères spéciaux Unicode provoquaient des `UnicodeEncodeError`.
   - *Solution :* Configuration automatique de `sys.stdout.reconfigure(encoding="utf-8")` et standardisation des logs d'affichage.
2. **Sécurité de sérialisation skops / MLflow 3.16+ :**
   - *Problème :* Les versions récentes de MLflow auditent les objets Scikit-Learn via `skops` et bloquent par défaut `sklearn.tree._tree.Tree` pour des motifs de sécurité.
   - *Solution :* Ajout du paramètre `skops_trusted_types=["sklearn.tree._tree.Tree"]` lors de l'appel à `mlflow.sklearn.log_model`.
