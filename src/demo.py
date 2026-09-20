"""
Script de démonstration réelle : testez avec vos propres mesures !
Permet de tester 3 cas réels ou de saisir vos propres mesures.
"""

import sys
import warnings
import joblib
import pandas as pd

warnings.filterwarnings("ignore")

# Encodage Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

CLASS_NAMES = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
DESCRIPTIONS = [
    "Petite fleur sauvage aux pétales très courts et discrets.",
    "Fleur de taille moyenne, la plus répandue dans les prairies.",
    "Grande fleur robuste aux larges pétales colorés."
]


def test_flower(sepal_len, sepal_wid, petal_len, petal_wid, name="Fleur inconnue"):
    model = joblib.load("models/iris_model.pkl")

    sample_df = pd.DataFrame(
        [[sepal_len, sepal_wid, petal_len, petal_wid]],
        columns=FEATURE_NAMES
    )

    pred_idx = model.predict(sample_df)[0]
    probas = model.predict_proba(sample_df)[0]
    confidence = probas[pred_idx] * 100

    print("-" * 65)
    print(f"🌸 TEST : {name}")
    print(f"   Mesures saisies : Sépale={sepal_len}x{sepal_wid} cm | Pétale={petal_len}x{petal_wid} cm")
    print(f"   ➜ Espèce prédite : *** {CLASS_NAMES[pred_idx].upper()} *** (Classe {pred_idx})")
    print(f"   ➜ Indice de certitude : {confidence:.1f}%")
    print(f"   ➜ Description : {DESCRIPTIONS[pred_idx]}")
    print(f"   ➜ Détail des probabilités :")
    for c_name, p in zip(CLASS_NAMES, probas):
        print(f"      - {c_name:17} : {p * 100:.1f}%")


def main():
    print("=" * 65)
    print("🧪 DÉMONSTRATION EN SITUATION RÉELLE : PRÉDICTION SUR 3 FLEURS")
    print("=" * 65)

    # 1. Petite fleur trouvée près d'un ruisseau
    test_flower(
        sepal_len=5.0, sepal_wid=3.4,
        petal_len=1.5, petal_wid=0.2,
        name="Fleur 1 (Petite fleur)"
    )

    # 2. Fleur moyenne trouvée dans un champ
    test_flower(
        sepal_len=6.0, sepal_wid=2.8,
        petal_len=4.5, petal_wid=1.4,
        name="Fleur 2 (Fleur moyenne de champ)"
    )

    # 3. Grande fleur trouvée dans une serre
    test_flower(
        sepal_len=7.2, sepal_wid=3.2,
        petal_len=6.0, petal_wid=2.3,
        name="Fleur 3 (Grande fleur de serre)"
    )

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
