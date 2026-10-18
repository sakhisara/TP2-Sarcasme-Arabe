
import re
import unicodedata
from collections import Counter
from pathlib import Path
import ast
from .config import PAD_ID, UNK_ID, MODELES_E

def labels_binaires(df):
    labels = (
        df["sarcasm"].astype(str).str.strip().str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
    )

    if labels.isna().any():
        raise ValueError("Valeur inattendue dans la colonne sarcasm.")

    return labels.astype(int)

def guillemets_externes(texte):
    texte = str(texte).strip()
    return (
        len(texte) >= 2
        and texte.startswith('"')
        and texte.endswith('"')
    )

def normaliser_comparaison(texte):
    texte = unicodedata.normalize("NFC", str(texte)).strip()

    # Retirer une paire de guillemets externes
    if guillemets_externes(texte):
        texte = texte[1:-1]

    texte = re.sub(r"[\u064B-\u065F\u0670]", "", texte)
    texte = texte.replace("\u0640", "")
    texte = re.sub(r"[أإآٱ]", "ا", texte)
    texte = re.sub(r"\s+", " ", texte).strip()

    return texte

def nettoyer_baseline(texte):
    texte = str(texte).strip()

    if guillemets_externes(texte):
        texte = texte[1:-1]

    return re.sub(r"\s+", " ", texte).strip()

def tokeniser_c(texte):
    texte_normalise = normaliser_comparaison(texte)

    # Mots et nombres ; ponctuations et symboles conservés séparément
    return re.findall(r"\w+|[^\w\s]", texte_normalise)

def retirer_guillemets_f(texte):
    texte = str(texte).strip()

    if (
        len(texte) >= 2
        and texte.startswith('"')
        and texte.endswith('"')
    ):
        texte = texte[1:-1]

    return texte

def retirer_raccourcis_f(texte):
    texte = retirer_guillemets_f(texte)

    texte = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        texte
    )

    texte = re.sub(r"@\w+", " ", texte)

    return re.sub(r"\s+", " ", texte).strip()

def tokens_vers_ids(tokens, vocabulaire):
    return [
        vocabulaire.get(token, UNK_ID)
        for token in tokens
    ]

def textes_pour_e(df, nom_court):
    from arabert.preprocess import ArabertPreprocessor

    textes = []

    prep = (
        ArabertPreprocessor(
            model_name=MODELES_E[nom_court]
        )
        if nom_court == "AraBERT"
        else None
    )

    for texte in df["tweet"].astype(str):
        texte = texte.strip()

        if (
            len(texte) >= 2
            and texte.startswith('"')
            and texte.endswith('"')
        ):
            texte = texte[1:-1]

        if prep is not None:
            texte = prep.preprocess(texte)
        else:
            texte = " ".join(texte.split())

        textes.append(texte)

    return textes

def preparer_texte_d(texte, preprocesseur):
    """Retrait des guillemets puis préprocesseur AraBERT fourni par l'appelant."""
    return preprocesseur.preprocess(retirer_guillemets_f(texte))

def construire_vocabulaire(sequences_train, frequence_min=2):
    """Fournir uniquement les tokens de TRAIN, dans leur ordre d'origine."""
    if frequence_min < 1:
        raise ValueError("La fréquence minimale doit être positive.")
    frequences = Counter(t for sequence in sequences_train for t in sequence)
    vocabulaire = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    for token, frequence in sorted(frequences.items()):
        if frequence >= frequence_min and token not in vocabulaire:
            vocabulaire[token] = len(vocabulaire)
    return vocabulaire

def taux_oov(sequences, vocabulaire):
    total = inconnus = 0
    for sequence in sequences:
        for token in sequence:
            total += 1
            inconnus += token not in vocabulaire
    return {"total_tokens": total, "tokens_OOV": inconnus,
            "taux_OOV_pct": 100 * inconnus / total if total else 0.0}

def charger_clean_str_aravec(chemin_utilities):
    """Extraire clean_str du utilities.py officiel déjà téléchargé, comme en C.

    L'appel est explicite ; aucun téléchargement n'est déclenché ici.
    """
    chemin = Path(chemin_utilities)
    arbre = ast.parse(chemin.read_text(encoding="utf-8-sig"))
    fonction = next(n for n in arbre.body
                    if isinstance(n, ast.FunctionDef) and n.name == "clean_str")
    espace = {"re": re}
    exec(compile(ast.Module(body=[fonction], type_ignores=[]), str(chemin), "exec"), espace)
    return espace["clean_str"]

def retirer_emojis_f(texte):
    import emoji
    texte = str(texte)
    if hasattr(emoji, "emoji_list"):
        for element in reversed(emoji.emoji_list(texte)):
            texte = texte[:element["match_start"]] + " " + texte[element["match_end"]:]
        return texte
    return emoji.get_emoji_regexp().sub(" ", texte)

def retirer_marqueurs_f(texte):
    texte = retirer_emojis_f(str(texte))

    # Même marqueur de rire arabe que dans l'audit.
    texte = re.sub(r"ه{3,}", " ", texte)

    # Supprimer les groupes de ponctuation identique répétée.
    texte = re.sub(
        r"([!?؟.,،؛:…])\1+",
        " ",
        texte
    )

    return re.sub(r"\s+", " ", texte).strip()
