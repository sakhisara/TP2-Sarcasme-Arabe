"""Chargement, split imposé et nettoyage ; aucune modification des données de l'appelant."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocess import (labels_binaires, normaliser_comparaison, tokeniser_c,
                         tokens_vers_ids, construire_vocabulaire)

BASE = "https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/"

def separer_training(full):
    """90 % / 10 %, graine 42 et stratification sur sarcasm, comme le TP."""
    return train_test_split(full, test_size=0.10, random_state=42,
                            stratify=full["sarcasm"])

def charger_donnees(base=BASE):
    """base : URL du dossier officiel ou chemin d'un dossier contenant les deux CSV."""
    base = str(base).rstrip("/") + "/"
    full = pd.read_csv(base + "training_data.csv")
    test = pd.read_csv(base + "testing_data.csv")
    full["id"] = range(len(full))
    test["id"] = range(len(test))
    train, val = separer_training(full)
    return train.copy(), val.copy(), test

def nettoyer_train(train_brut, test_brut):
    """Politique de la cellule 24, dans le même ordre.

    1. Chevauchements avec test (texte seulement).
    2. Groupes contradictoires internes à train restant.
    3. Répétitions restantes (première occurrence conservée).
    Les labels du test n'interviennent pas dans les suppressions.
    Retour : train nettoyé, bilan, détail des contradictions retirées de train.
    """
    colonnes = list(train_brut.columns)
    travail = train_brut.copy()
    initial = len(travail)
    travail["texte_normalise"] = travail["tweet"].apply(normaliser_comparaison)
    travail["label_binaire"] = labels_binaires(travail)
    normalises_test = test_brut["tweet"].apply(normaliser_comparaison)
    cles_test = set(normalises_test[normalises_test.ne("")])
    chevauchement = travail["tweet"].isin(set(test_brut["tweet"])) | (
        travail["texte_normalise"].ne("") & travail["texte_normalise"].isin(cles_test))
    n_chevauchements = int(chevauchement.sum())
    travail = travail.loc[~chevauchement].copy()
    diversite = travail.groupby("texte_normalise")["label_binaire"].nunique()
    cles = diversite[(diversite > 1) & (diversite.index != "")].index
    masque = travail["texte_normalise"].isin(cles)
    conflits = travail.loc[masque].copy()
    n_conflits = int(masque.sum())
    travail = travail.loc[~masque].copy()
    repetitions = travail["texte_normalise"].ne("") & travail.duplicated(
        "texte_normalise", keep="first")
    n_repetitions = int(repetitions.sum())
    resultat = travail.loc[~repetitions, colonnes].copy()
    assert initial == len(resultat) + n_chevauchements + n_conflits + n_repetitions
    bilan = {"train_initial": initial, "retires_chevauchement_test": n_chevauchements,
             "retires_contradictions_train": n_conflits,
             "retires_repetitions_train": n_repetitions, "train_final": len(resultat)}
    return resultat, bilan, conflits

def preparer_donnees_c(train, val, test, frequence_min=2):
    """Le vocabulaire et la longueur sont ajustés sur train uniquement."""
    copies = [df.copy() for df in (train, val, test)]
    for df in copies:
        df["tokens"] = df["tweet"].apply(tokeniser_c)
    vocab = construire_vocabulaire(copies[0]["tokens"], frequence_min)
    for df in copies:
        df["input_ids"] = df["tokens"].apply(lambda tokens: tokens_vers_ids(tokens, vocab))
    if copies[0].empty:
        raise ValueError("Train est vide.")
    max_len = max(1, min(256, int(np.ceil(np.quantile(
        copies[0]["input_ids"].apply(len), 0.95)))))
    return (*copies, vocab, max_len)
