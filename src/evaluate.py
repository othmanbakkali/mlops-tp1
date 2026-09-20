"""
Module d'evaluation du modele entraine sur le jeu de test.
Genere le rapport de classification detaille, la matrice de confusion
et sauvegarde les metriques dans experiments/.
"""

import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Encodage Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def evaluate_model(
    model_path="models/iris_model.pkl",
    test_data_path="data/processed/test.csv",
    output_dir="experiments"
):
    print("=" * 60)
    print("[ETAPE] EVALUATION DU MODELE (evaluate.py)")
    print("=" * 60)

    # Verification des fichiers
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modele introuvable : {model_path}. Executez src/train.py d'abord.")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Donnees de test introuvables : {test_data_path}. Executez src/prepare.py d'abord.")

    # 1. Chargement du modele et des donnees de test
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)

    feature_cols = [c for c in test_df.columns if c != "target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    # 2. Predictions
    y_pred = model.predict(X_test)

    # 3. Metriques
    acc = accuracy_score(y_test, y_pred)
    target_names = ["setosa", "versicolor", "virginica"]
    clf_report = classification_report(y_test, y_pred, target_names=target_names)

    print(f"Modele evalue : {model_path}")
    print(f"Jeu de test   : {X_test.shape[0]} observations\n")
    print(f"[METRIQUE] ACCURACY : {acc:.4f} ({acc * 100:.2f}%)\n")
    print("[RAPPORT DE CLASSIFICATION] :")
    print(clf_report)

    # 4. Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :")
    print(cm)

    # Sauvegarde des resultats
    os.makedirs(output_dir, exist_ok=True)

    report_file = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== RAPPORT D'EVALUATION MLOPS TP1 ===\n\n")
        f.write(f"Accuracy : {acc:.4f}\n\n")
        f.write("Classification Report :\n")
        f.write(clf_report)
        f.write("\nMatrice de confusion :\n")
        f.write(str(cm))
        f.write("\n")
    print(f"\n[OK] Rapport texte sauvegarde dans : {report_file}")

    # Visualisation de la matrice de confusion
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Matrice de Confusion — Modèle Iris (Random Forest)")
    plt.tight_layout()

    cm_img_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_img_path, dpi=150)
    plt.close()
    print(f"[OK] Matrice de confusion sauvegardee dans : {cm_img_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate_model()
