"""
Module d'analyse du Data Drift (drift.py)
Analyse les statistiques descriptives de base (baseline d'entraînement)
et permet de comparer avec des distributions en production pour détecter des dérives.
"""

import sys
import pandas as pd
from sklearn.datasets import load_iris

# Encodage Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    print("=" * 65)
    print("[ACTIVITÉ DATA DRIFT] STATISTIQUES DESCRIPTIVES DU DATASET INITIAL")
    print("=" * 65)

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    print(df.describe())
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
