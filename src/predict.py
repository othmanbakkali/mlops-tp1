"""
Module d'inférence (predict.py)
Charge le modèle sérialisé iris_model.pkl et effectue une prédiction.
"""

import sys
import warnings
import joblib
from sklearn.datasets import load_iris

warnings.filterwarnings("ignore")

# Encodage Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    # 1. Charger le modèle
    model = joblib.load("models/iris_model.pkl")

    # 2. Charger les données Iris
    iris = load_iris()
    sample = [iris.data[0]]

    # 3. Prédiction
    prediction = model.predict(sample)

    # 4. Affichage du résultat
    print("Classe :", prediction[0])
    print("Nom :", iris.target_names[prediction[0]])


if __name__ == "__main__":
    main()
