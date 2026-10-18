
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, precision_recall_curve, auc)
from .preprocess import normaliser_comparaison, labels_binaires

def mesurer_c(y, scores, seuil):
    predictions = (scores >= seuil).astype(int)

    precision, rappel, _ = precision_recall_curve(
        y, scores
    )

    return {
        "val_accuracy": accuracy_score(y, predictions),
        "val_precision_sarc": precision_score(
            y, predictions, zero_division=0
        ),
        "val_rappel_sarc": recall_score(
            y, predictions, zero_division=0
        ),
        "val_f1_sarc": f1_score(
            y, predictions, zero_division=0
        ),
        "val_macro_f1": f1_score(
            y, predictions,
            average="macro",
            zero_division=0
        ),
        "val_pr_auc": auc(
            rappel[::-1],
            precision[::-1]
        )
    }

def choisir_seuil_c(y, scores):
    precision, rappel, seuils = precision_recall_curve(
        y, scores
    )

    f1 = (
        2 * precision[:-1] * rappel[:-1]
        / np.maximum(
            precision[:-1] + rappel[:-1],
            1e-12
        )
    )

    return float(seuils[np.argmax(f1)])

def metriques_d(y, scores, seuil):
    predictions = (scores >= seuil).astype(int)

    precision, rappel, _ = precision_recall_curve(
        y, scores
    )

    return {
        "val_accuracy": accuracy_score(y, predictions),
        "val_precision_sarc": precision_score(
            y, predictions, zero_division=0
        ),
        "val_rappel_sarc": recall_score(
            y, predictions, zero_division=0
        ),
        "val_f1_sarc": f1_score(
            y, predictions, zero_division=0
        ),
        "val_macro_f1": f1_score(
            y, predictions,
            average="macro",
            zero_division=0
        ),
        "val_pr_auc": auc(
            rappel[::-1],
            precision[::-1]
        )
    }

def seuil_optimal_d(y, scores):
    precision, rappel, seuils = precision_recall_curve(
        y, scores
    )

    f1 = (
        2 * precision[:-1] * rappel[:-1]
        / np.maximum(
            precision[:-1] + rappel[:-1],
            1e-12
        )
    )

    return float(seuils[np.argmax(f1)])

def metriques_f(y, predictions):
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision_sarc": precision_score(
            y, predictions, zero_division=0
        ),
        "rappel_sarc": recall_score(
            y, predictions, zero_division=0
        ),
        "f1_sarc": f1_score(
            y, predictions, zero_division=0
        ),
        "macro_f1": f1_score(
            y, predictions,
            average="macro",
            zero_division=0
        )
    }

def f1_rapide_f(y, prediction):
    tp = np.sum((y == 1) & (prediction == 1))
    fp = np.sum((y == 0) & (prediction == 1))
    fn = np.sum((y == 1) & (prediction == 0))

    denominateur = 2 * tp + fp + fn

    return 2 * tp / denominateur if denominateur else 0.0

def bootstrap_f1_f(y, prediction, n_boot=1000):
    rng = np.random.default_rng(42)
    valeurs = []

    for _ in range(n_boot):
        indices = rng.integers(
            0, len(y), size=len(y)
        )

        valeurs.append(
            f1_rapide_f(
                y[indices],
                prediction[indices]
            )
        )

    return np.percentile(valeurs, [2.5, 97.5])


def verifier_predictions(df):
    requis = {"id", "y_true", "y_pred"}
    if not requis.issubset(df.columns):
        raise ValueError("Colonnes attendues : id, y_true, y_pred.")
    if df.empty or not df["id"].is_unique or df["id"].isna().any():
        raise ValueError("Prédictions vides ou identifiants invalides/répétés.")
    for colonne in ("y_true", "y_pred"):
        if not df[colonne].isin([0, 1]).all():
            raise ValueError("Les labels doivent être 0 ou 1.")
    return df.set_index("id").sort_index()

def mcnemar_predictions(a, b):
    """Test exact des taux d'erreur, aligné par id (partie F5)."""
    a, b = verifier_predictions(a), verifier_predictions(b)
    if not a.index.equals(b.index) or not np.array_equal(a["y_true"], b["y_true"]):
        raise ValueError("Les fichiers doivent contenir les mêmes id et vrais labels.")
    ca = (a["y_pred"] == a["y_true"]).to_numpy()
    cb = (b["y_pred"] == b["y_true"]).to_numpy()
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = n01 + n10
    p = float(binomtest(n01, n=n, p=0.5).pvalue) if n else 1.0
    return {"nombre_tweets": len(a), "A_correct_B_incorrect": n01,
            "A_incorrect_B_correct": n10, "p_value": p,
            "significatif_5pct": p < 0.05}

def croiser_erreurs_conflits(erreurs, train_avant_nettoyage, test):
    """F6 : audit étendu train brut + test, sans modifier les annotations manuelles.

    Ce périmètre est plus large que les contradictions internes retirées en 4.1.
    Les labels du test servent uniquement à cette analyse a posteriori.
    """
    audit = pd.concat([
        train_avant_nettoyage.assign(ensemble="train_avant_nettoyage"),
        test.assign(ensemble="test")
    ], ignore_index=True)
    audit["texte_normalise"] = audit["tweet"].apply(normaliser_comparaison)
    audit["label_binaire"] = labels_binaires(audit)
    diversite = audit.loc[audit["texte_normalise"].ne("")].groupby(
        "texte_normalise")["label_binaire"].nunique()
    cles = set(diversite[diversite > 1].index)
    conflits = audit.loc[audit["texte_normalise"].isin(cles)].copy()
    resultat = erreurs.copy()
    resultat["doublon_contradictoire_4_1"] = resultat["tweet"].apply(
        normaliser_comparaison).isin(cles)
    return resultat, conflits

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Évaluer un CSV existant sans entraîner.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--comparer", type=Path, help="Second CSV pour McNemar.")
    args = parser.parse_args()
    a = pd.read_csv(args.csv)
    propre = verifier_predictions(a)
    print(metriques_f(propre["y_true"].to_numpy(), propre["y_pred"].to_numpy()))
    if args.comparer:
        print(mcnemar_predictions(a, pd.read_csv(args.comparer)))

if __name__ == "__main__":
    main()
