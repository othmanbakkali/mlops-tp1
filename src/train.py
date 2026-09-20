"""
Module d'entrainement du modele avec suivi d'experiences MLflow.
Definit la fonction train_model() et permet de comparer plusieurs configurations d'hyperparametres.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Encodage Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_processed_data(data_dir="data"):
    """Charge les donnees traitees depuis data/processed/."""
    train_path = os.path.join(data_dir, "processed", "train.csv")
    test_path = os.path.join(data_dir, "processed", "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Fichiers d'entrainement/test introuvables dans {data_dir}/processed/. "
            "Veuillez executer src/prepare.py d'abord."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    return X_train, X_test, y_train, y_test


def train_model(
    n_estimators=100,
    random_state=42,
    experiment_name="TP1_Iris_MLOps",
    model_save_path="models/iris_model.pkl",
    data_dir="data"
):
    """
    Entraine un RandomForestClassifier, evalue ses performances,
    enregistre l'experience dans MLflow et sauvegarde le modele sur le disque.
    """
    print(f"\n[INFO] Lancement de l'experience : n_estimators={n_estimators}, random_state={random_state}")

    # Chargement des donnees
    X_train, X_test, y_train, y_test = load_processed_data(data_dir=data_dir)

    # Definition de l'experience MLflow
    mlflow.set_experiment(experiment_name)

    run_name = f"RF_n_estimators_{n_estimators}"
    with mlflow.start_run(run_name=run_name):
        # 1. Instanciation et entrainement du modele
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # 2. Predictions
        y_pred = model.predict(X_test)

        # 3. Calcul des metriques d'evaluation
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # 4. Logging dans MLflow (Defi 3)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log du modele MLflow (compatible MLflow v3 et skops)
        try:
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
        except TypeError:
            # Fallback pour versions antérieures de MLflow
            mlflow.sklearn.log_model(model, "model")

        print("  Resultats du Run :")
        print(f"    - Accuracy  : {acc:.4f}")
        print(f"    - Precision : {prec:.4f}")
        print(f"    - Recall    : {rec:.4f}")
        print(f"    - F1-score  : {f1:.4f}")

    # 5. Sauvegarde locale du modele serialise
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"  [OK] Modele serialise sauvegarde dans : {model_save_path}")

    return {
        "n_estimators": n_estimators,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "model": model
    }


def run_benchmark():
    """Defi 4 : Tester les configurations n_estimators = 50, 100, 200 et comparer."""
    print("=" * 60)
    print("[DEFI 4] BENCHMARK MULTI-CONFIGURATIONS (50, 100, 200)")
    print("=" * 60)

    configs = [50, 100, 200]
    results = []

    for n in configs:
        res = train_model(n_estimators=n, random_state=42)
        results.append(res)

    print("\n" + "=" * 60)
    print("[RESULTAT] TABLEAU COMPARATIF DES EXPERIENCES (MLflow)")
    print("=" * 60)
    summary_df = pd.DataFrame([
        {
            "Run": f"RF_n_estimators_{r['n_estimators']}",
            "n_estimators": r["n_estimators"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "F1-score": f"{r['f1_score']:.4f}",
        }
        for r in results
    ])
    print(summary_df.to_string(index=False))

    # Sauvegarde du resume dans experiments/
    os.makedirs("experiments", exist_ok=True)
    summary_df.to_csv("experiments/benchmark_results.csv", index=False)
    print(f"\n[OK] Resultats sauvegardes dans : experiments/benchmark_results.csv")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrainement du modele Iris avec MLflow")
    parser.add_argument("--n_estimators", type=int, default=None, help="Nombre d'arbres")
    parser.add_argument("--benchmark", action="store_true", help="Executer le benchmark (50, 100, 200)")

    args = parser.parse_args()

    if args.benchmark or args.n_estimators is None:
        run_benchmark()
    else:
        train_model(n_estimators=args.n_estimators)
