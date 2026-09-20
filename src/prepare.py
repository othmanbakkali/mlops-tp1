"""
Module de preparation des donnees pour le projet MLOps TP1.
Charge le dataset Iris, applique un decoupage stratifie train/test,
et sauvegarde les donnees preparees.
"""

import os
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Gestion de l'encodage pour Windows PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def prepare_data(data_dir="data", test_size=0.2, random_state=42):
    print("=" * 60)
    print("[ETAPE] PREPARATION DES DONNEES (prepare.py)")
    print("=" * 60)

    # 1. Chargement des donnees Iris
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    print("Dataset charge avec succes :")
    print(f"  - Nombre total d'observations : {X.shape[0]}")
    print(f"  - Nombre de variables d'entree : {X.shape[1]}")
    print(f"  - Variables : {list(X.columns)}")
    print(f"  - Classes cibles : {list(iris.target_names)} (labels: {sorted(y.unique())})\n")

    # 2. Sauvegarde des donnees brutes
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_df = X.copy()
    raw_df["target"] = y
    raw_path = os.path.join(raw_dir, "iris_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"[OK] Donnees brutes sauvegardees dans : {raw_path}")

    # 3. Separation Train / Test stratifiee
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\nSeparation Train/Test effectuee (test_size={test_size}, random_state={random_state}) :")
    print(f"  - Train set : {X_train.shape[0]} exemples ({X_train.shape[1]} features)")
    print(f"  - Test set  : {X_test.shape[0]} exemples ({X_test.shape[1]} features)")

    # 4. Sauvegarde des datasets d'entrainement et de test
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] Train dataset sauvegarde dans : {train_path}")
    print(f"[OK] Test dataset sauvegarde dans  : {test_path}")
    print("=" * 60 + "\n")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_data()
