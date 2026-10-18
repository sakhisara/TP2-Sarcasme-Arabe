"""Entraînements extraits et paramétrés ; rien n'est entraîné à l'import.

Les nouveaux fichiers sont écrits par défaut dans experiences_src/, séparément
des résultats historiques. L'import du module n'entraîne et ne télécharge rien.
"""
import gc
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from .config import MODELES_E, SENTIMENTS_E, CONFIG_E as CONFIG_E_DEFAUT
from .datasets_torch import creer_loaders_c, creer_loaders_e
from .models import ModeleC, ModeleE
from .evaluate import mesurer_c, choisir_seuil_c, metriques_d, seuil_optimal_d

def fixer_graine_c(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def predire_c(modele, loader):
    DEVICE_C = next(modele.parameters()).device
    modele.eval()

    vrais_labels = []
    probabilites = []

    for ids, longueurs, labels in loader:
        logits = modele(
            ids.to(DEVICE_C),
            longueurs
        )

        scores = torch.softmax(
            logits,
            dim=-1
        )[:, 1]

        probabilites.extend(
            scores.cpu().tolist()
        )

        vrais_labels.extend(
            labels.tolist()
        )

    return (
        np.asarray(vrais_labels),
        np.asarray(probabilites)
    )

def regler_embedding_c(modele, config, epoque):
    mode = config["embedding"]

    entrainable = (
        mode == "random"
        or (
            mode == "aravec_thaw"
            and epoque > config["freeze_epochs"]
        )
    )

    modele.embedding.weight.requires_grad_(
        entrainable
    )

def ajouter_resultat_c(resultat, chemin):
    ligne = pd.DataFrame([{
        cle: valeur
        for cle, valeur in resultat.items()
        if cle != "config"
    }])

    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    if chemin.exists():
        ligne = pd.concat(
            [pd.read_csv(chemin), ligne],
            ignore_index=True
        )

    ligne.to_csv(chemin, index=False)

def entrainer_c(
    config,
    matrice_embedding,
    train_df,
    val_df,
    vocab,
    device=None,
    output_dir="experiences_src"
):
    DEVICE_C = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    for sous_dossier in ("results", "predictions", "checkpoints"):
        (output_dir / sous_dossier).mkdir(parents=True, exist_ok=True)

    config = dict(config)
    fixer_graine_c(config["seed"])

    train_loader, val_loader, effectifs = creer_loaders_c(
        train_df,
        val_df,
        config
    )

    modele = ModeleC(
        matrice_embedding,
        config
    ).to(DEVICE_C)

    poids_classes = None

    if config["methode"] == "weighted":
        poids_classes = torch.tensor(
            effectifs.sum() / (2 * effectifs),
            dtype=torch.float32,
            device=DEVICE_C
        )

    critere = nn.CrossEntropyLoss(
        weight=poids_classes
    )

    # Tous les paramètres sont inscrits dans l'optimiseur,
    # y compris les embeddings qui pourront être dégelés.
    optimiseur = torch.optim.Adam(
        modele.parameters(),
        lr=config["lr"]
    )

    nom = (
        f"{config['arch']}_{config['embedding']}"
        f"_pos{int(config['positions'])}"
        f"_{config['methode']}_seed{config['seed']}"
    )

    meilleur_f1 = -1.0
    meilleur_etat = None
    meilleure_epoque = 0
    attente = 0
    historique = []
    max_parametres_entraines = 0

    if DEVICE_C.type == "cuda":
        torch.cuda.synchronize()

    debut = time.perf_counter()

    for epoque in range(1, config["max_epochs"] + 1):
        regler_embedding_c(
            modele,
            config,
            epoque
        )

        nombre_parametres = sum(
            p.numel()
            for p in modele.parameters()
            if p.requires_grad
        )

        max_parametres_entraines = max(
            max_parametres_entraines,
            nombre_parametres
        )

        modele.train()
        somme_pertes = 0.0
        nombre_exemples = 0

        for ids, longueurs, labels in train_loader:
            ids = ids.to(DEVICE_C)
            labels = labels.to(DEVICE_C)

            optimiseur.zero_grad(set_to_none=True)

            logits = modele(ids, longueurs)
            perte = critere(logits, labels)

            perte.backward()

            nn.utils.clip_grad_norm_(
                modele.parameters(),
                max_norm=1.0
            )

            optimiseur.step()

            somme_pertes += perte.item() * len(labels)
            nombre_exemples += len(labels)

        y, scores = predire_c(
            modele,
            val_loader
        )

        f1_val = mesurer_c(
            y, scores, seuil=0.5
        )["val_f1_sarc"]

        perte_moyenne = somme_pertes / nombre_exemples

        historique.append({
            "epoque": epoque,
            "perte_train": perte_moyenne,
            "val_f1_seuil_05": f1_val,
            "parametres_entrainables": nombre_parametres
        })

        print(
            f"{nom} | époque {epoque} | "
            f"perte {perte_moyenne:.4f} | "
            f"F1 val {f1_val:.4f}"
        )

        # Pour le mode dégelé, sélectionner parmi les
        # checkpoints obtenus APRÈS le dégel.
        selection_autorisee = (
            config["embedding"] != "aravec_thaw"
            or epoque > config["freeze_epochs"]
        )

        if selection_autorisee:
            if f1_val > meilleur_f1 + 1e-8:
                meilleur_f1 = f1_val
                meilleure_epoque = epoque
                attente = 0

                meilleur_etat = {
                    cle: valeur.detach().cpu().clone()
                    for cle, valeur in modele.state_dict().items()
                }

            else:
                attente += 1

                if attente >= config["patience"]:
                    print("Early stopping.")
                    break

    if meilleur_etat is None:
        raise ValueError(
            "Aucun checkpoint sélectionné : "
            "vérifier max_epochs et freeze_epochs."
        )

    if DEVICE_C.type == "cuda":
        torch.cuda.synchronize()

    temps_entrainement = time.perf_counter() - debut

    modele.load_state_dict(meilleur_etat)

    y, scores = predire_c(
        modele,
        val_loader
    )

    seuil_optimise = choisir_seuil_c(y, scores)

    seuil = (
        seuil_optimise
        if config["methode"] == "threshold"
        else 0.5
    )

    chemin_checkpoint = (output_dir / "checkpoints") / f"{nom}.pt"

    torch.save({
        "state_dict": meilleur_etat,
        "config": config,
        "vocabulaire": vocab,
        "embedding_shape": list(matrice_embedding.shape),
        "seuil": seuil,
        "meilleure_epoque": meilleure_epoque
    }, chemin_checkpoint)

    resultat = {
        "modele": nom,
        "arch": config["arch"],
        "embedding": config["embedding"],
        "positions": config["positions"],
        "methode": config["methode"],
        "graine": config["seed"],
        "pretraitement": (
            "normalisation C + caracteres"
            if config["arch"] == "C4"
            else "normalisation C + mots et symboles"
        ),
        "seuil": seuil,
        "seuil_optimal_val": seuil_optimise,
        "temps_entrainement_s": temps_entrainement,
        "parametres_entraines": max_parametres_entraines,
        "meilleure_epoque": meilleure_epoque,
        "epoques_executees": len(historique),
        "taille_checkpoint_Mo": (
            chemin_checkpoint.stat().st_size / 1e6
        ),
        "checkpoint": str(chemin_checkpoint),
        "config": config,
        **mesurer_c(y, scores, seuil)
    }

    pd.DataFrame(historique).to_csv(
        output_dir / "results" / f"{nom}_historique.csv",
        index=False
    )

    pd.DataFrame({
        "id": val_df["id"].to_numpy(),
        "y_true": y,
        "y_pred": (scores >= seuil).astype(int),
        "prob_sarcastique": scores
    }).to_csv(
        output_dir / "predictions" / f"{nom}_val.csv",
        index=False
    )

    ajouter_resultat_c(
        resultat,
        output_dir / "results/results.csv"
    )

    ajouter_resultat_c(
        resultat,
        output_dir / "results/C_results.csv"
    )

    del modele, optimiseur
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return resultat

@torch.no_grad()
def evaluer_e(modele, loader):
    DEVICE_E = next(modele.parameters()).device
    AMP_E = DEVICE_E.type == "cuda"
    modele.eval()

    vrais_labels = []
    scores = []

    for entrees, y_sarc, _ in loader:
        entrees = {
            cle: valeur.to(DEVICE_E)
            for cle, valeur in entrees.items()
        }

        with torch.autocast(
            device_type=DEVICE_E.type,
            dtype=torch.float16,
            enabled=AMP_E
        ):
            logits_sarc, _ = modele(**entrees)

        probabilites = torch.softmax(
            logits_sarc.float(),
            dim=-1
        )[:, 1]

        scores.extend(probabilites.cpu().tolist())
        vrais_labels.extend(y_sarc.tolist())

    y = np.array(vrais_labels)
    scores = np.array(scores)

    seuil = seuil_optimal_d(y, scores)

    return (
        y,
        scores,
        seuil,
        metriques_d(y, scores, seuil)
    )

def entrainer_e(
    nom_court,
    multitache=False,
    lam=0.0,
    graine=42,
    *,
    train_df,
    val_df,
    config=None,
    device=None,
    output_dir="experiences_src"
):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    train, val = train_df, val_df
    CONFIG_E = {**CONFIG_E_DEFAUT, **(config or {})}
    DEVICE_E = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    AMP_E = DEVICE_E.type == "cuda"
    output_dir = Path(output_dir)
    for sous_dossier in ("results", "predictions", "checkpoints/E"):
        (output_dir / sous_dossier).mkdir(parents=True, exist_ok=True)
    fixer_graine_c(graine)

    nom_modele = MODELES_E[nom_court]
    tache = "E2" if multitache else "E1"

    if not multitache:
        lam = 0.0

    nom_run = (
        f"{tache}_{nom_court}"
        f"_lambda{lam}_seed{graine}"
    )

    dossier_run = (output_dir / "checkpoints/E") / nom_run
    dossier_run.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        nom_modele
    )

    train_loader, val_loader, y_train = creer_loaders_e(
        nom_court,
        tokenizer,
        graine, train_df, val_df, CONFIG_E
    )

    modele = ModeleE(
        nom_modele,
        multitache=multitache, dropout=CONFIG_E["dropout"]
    ).to(DEVICE_E)

    effectifs = np.bincount(
        y_train,
        minlength=2
    )

    if (effectifs == 0).any():
        raise ValueError("Train doit contenir les deux classes.")

    poids = torch.tensor(
        effectifs.sum() / (2 * effectifs),
        dtype=torch.float32,
        device=DEVICE_E
    )

    # Ne pas appliquer le weight decay aux biais et LayerNorm.
    sans_decay = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight"
    )

    groupes = [
        {
            "params": [
                p for nom, p in modele.named_parameters()
                if not any(mot in nom for mot in sans_decay)
            ],
            "weight_decay": CONFIG_E["weight_decay"]
        },
        {
            "params": [
                p for nom, p in modele.named_parameters()
                if any(mot in nom for mot in sans_decay)
            ],
            "weight_decay": 0.0
        }
    ]

    optimiseur = torch.optim.AdamW(
        groupes,
        lr=CONFIG_E["lr"]
    )

    total_steps = (
        len(train_loader) * CONFIG_E["max_epochs"]
    )

    scheduler = get_linear_schedule_with_warmup(
        optimiseur,
        num_warmup_steps=math.ceil(
            CONFIG_E["warmup_ratio"] * total_steps
        ),
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=AMP_E
    )

    nombre_parametres = sum(
        p.numel()
        for p in modele.parameters()
        if p.requires_grad
    )

    meilleur_f1 = -1.0
    meilleure_epoque = 0
    attente = 0
    historique = []
    temps_train = 0.0

    chemin_checkpoint = dossier_run / "best.pt"

    for epoque in range(1, CONFIG_E["max_epochs"] + 1):
        modele.train()

        if AMP_E:
            torch.cuda.synchronize()

        debut_epoque = time.perf_counter()
        somme_pertes = 0.0

        for entrees, y_sarc, y_sent in train_loader:
            optimiseur.zero_grad(set_to_none=True)

            taille_batch = len(y_sarc)

            # Normalisation de la perte pondérée sur tout
            # le batch effectif de 32 exemples.
            denominateur_sarc = poids[
                y_sarc.to(DEVICE_E)
            ].sum()

            for debut in range(
                0,
                taille_batch,
                CONFIG_E["micro_batch"]
            ):
                fin = debut + CONFIG_E["micro_batch"]

                micro_entrees = {
                    cle: valeur[debut:fin].to(DEVICE_E)
                    for cle, valeur in entrees.items()
                }

                micro_sarc = y_sarc[debut:fin].to(DEVICE_E)
                micro_sent = y_sent[debut:fin].to(DEVICE_E)

                with torch.autocast(
                    device_type=DEVICE_E.type,
                    dtype=torch.float16,
                    enabled=AMP_E
                ):
                    logits_sarc, logits_sent = modele(
                        **micro_entrees
                    )

                    perte = F.cross_entropy(
                        logits_sarc.float(),
                        micro_sarc,
                        weight=poids,
                        reduction="sum"
                    ) / denominateur_sarc

                    if multitache:
                        perte_sent = F.cross_entropy(
                            logits_sent.float(),
                            micro_sent,
                            reduction="sum"
                        ) / taille_batch

                        perte = perte + lam * perte_sent

                scaler.scale(perte).backward()
                somme_pertes += perte.item()

            scaler.unscale_(optimiseur)

            nn.utils.clip_grad_norm_(
                modele.parameters(),
                max_norm=1.0
            )

            ancienne_echelle = scaler.get_scale()

            scaler.step(optimiseur)
            scaler.update()

            # Avancer le scheduler seulement si le pas
            # d'optimisation n'a pas été annulé en FP16.
            if scaler.get_scale() >= ancienne_echelle:
                scheduler.step()

        if AMP_E:
            torch.cuda.synchronize()

        temps_train += time.perf_counter() - debut_epoque

        y, scores, seuil, mesures = evaluer_e(
            modele,
            val_loader
        )

        f1_val = mesures["val_f1_sarc"]

        historique.append({
            "epoque": epoque,
            "perte_train": somme_pertes / len(train_loader),
            "seuil_val": seuil,
            **mesures
        })

        print(
            f"{nom_run} | époque {epoque} | "
            f"F1 val {f1_val:.4f} | seuil {seuil:.4f}"
        )

        if f1_val > meilleur_f1 + 1e-8:
            meilleur_f1 = f1_val
            meilleure_epoque = epoque
            attente = 0

            etat_cpu = {
                cle: valeur.detach().cpu()
                for cle, valeur in modele.state_dict().items()
            }

            torch.save({
                "state_dict": etat_cpu,
                "nom_modele": nom_modele,
                "revision_encodeur": getattr(
                    modele.enc.config,
                    "_commit_hash",
                    None
                ),
                "multitache": multitache,
                "lambda": lam,
                "graine": graine,
                "seuil": seuil,
                "config": CONFIG_E,
                "sentiment_mapping": SENTIMENTS_E,
                "meilleure_epoque": epoque
            }, chemin_checkpoint)

            del etat_cpu

        else:
            attente += 1

            if attente >= CONFIG_E["patience"]:
                print("Early stopping.")
                break

    # Restaurer le meilleur checkpoint.
    sauvegarde = torch.load(
        chemin_checkpoint,
        map_location="cpu",
        weights_only=False
    )

    modele.load_state_dict(
        sauvegarde["state_dict"]
    )

    seuil_retenu = sauvegarde["seuil"]
    del sauvegarde

    y, scores, _, _ = evaluer_e(
        modele,
        val_loader
    )

    resultat = {
        "modele": nom_run,
        "tache": tache,
        "encodeur": nom_court,
        "lambda": lam,
        "graine": graine,
        "seuil": seuil_retenu,
        "pretraitement": (
            "guillemets retires + ArabertPreprocessor"
            if nom_court == "AraBERT"
            else "guillemets retires + espaces harmonises"
        ),
        "temps_entrainement_s": temps_train,
        "parametres_entraines": nombre_parametres,
        "meilleure_epoque": meilleure_epoque,
        "checkpoint": str(chemin_checkpoint),
        **metriques_d(y, scores, seuil_retenu)
    }

    tokenizer.save_pretrained(
        dossier_run / "tokenizer"
    )

    pd.DataFrame(historique).to_csv(
        output_dir / "results" / f"{nom_run}_historique.csv",
        index=False
    )

    pd.DataFrame({
        "id": val["id"].to_numpy(),
        "y_true": y,
        "y_pred": (scores >= seuil_retenu).astype(int),
        "prob_sarcastique": scores
    }).to_csv(
        output_dir / "predictions" / f"{nom_run}_val.csv",
        index=False
    )

    ligne = pd.DataFrame([resultat])

    for chemin in [
        output_dir / "results/results.csv",
        output_dir / "results/E_results.csv"
    ]:
        if chemin.exists():
            journal = pd.concat(
                [pd.read_csv(chemin), ligne],
                ignore_index=True
            )
        else:
            journal = ligne

        journal.to_csv(chemin, index=False)

    del modele, optimiseur, scheduler, scaler
    gc.collect()

    if AMP_E:
        torch.cuda.empty_cache()

    return resultat


def entrainer_baseline(train_df, val_df, graine=42, output_dir="experiences_src"):
    """TF-IDF mots 1–2 + caractères 2–5 et LR balanced (partie A)."""
    import joblib
    from sklearn.pipeline import FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from .preprocess import nettoyer_baseline, labels_binaires
    dossier = Path(output_dir)
    for sous_dossier in ("results", "predictions", "checkpoints"):
        (dossier / sous_dossier).mkdir(parents=True, exist_ok=True)
    y_train = labels_binaires(train_df).to_numpy()
    y_val = labels_binaires(val_df).to_numpy()
    tfidf = FeatureUnion([
        ("mots", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=False)),
        ("caracteres", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False))
    ])
    debut = time.perf_counter()
    x_train = tfidf.fit_transform(train_df["tweet"].apply(nettoyer_baseline))
    x_val = tfidf.transform(val_df["tweet"].apply(nettoyer_baseline))
    lr = LogisticRegression(class_weight="balanced", solver="liblinear",
                            max_iter=2000, random_state=graine).fit(x_train, y_train)
    temps = time.perf_counter() - debut
    scores = lr.predict_proba(x_val)[:, list(lr.classes_).index(1)]
    seuil = choisir_seuil_c(y_val, scores)
    fichier = dossier / "checkpoints" / f"baseline_seed{graine}.joblib"
    sauvegarde = {"tfidf": tfidf, "classifieur": lr, "seuil": seuil}
    joblib.dump(sauvegarde, fichier)
    resultat = {"modele": "TF-IDF mots+caracteres + LR balanced", "graine": graine,
        "pretraitement": "sans guillemets externes + espaces harmonises", "seuil": seuil,
        "temps_entrainement_s": temps, "parametres_entraines": lr.coef_.size + lr.intercept_.size,
        "checkpoint": str(fichier), **mesurer_c(y_val, scores, seuil)}
    ajouter_resultat_c(resultat, dossier / "results/results.csv")
    pd.DataFrame({"id": val_df["id"].to_numpy(), "y_true": y_val,
        "y_pred": (scores >= seuil).astype(int), "prob_sarcastique": scores}).to_csv(
            dossier / "predictions" / f"baseline_seed{graine}_val.csv", index=False)
    return resultat


def entrainer_d1(x_train, x_val, train_df, val_df, representation="moyenne",
                 graine=42, temps_extraction=0.0, revision_encodeur=None,
                 output_dir="experiences_src"):
    """Régression logistique D1 sur des représentations déjà extraites.

    L'appelant fournit des matrices alignées avec les DataFrames correspondants.
    Le temps d'extraction commun peut être transmis pour le journal.
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from .preprocess import labels_binaires
    from .config import NOM_ARABERT
    if representation not in {"CLS", "moyenne"}:
        raise ValueError("Représentation attendue : CLS ou moyenne.")
    if len(x_train) != len(train_df) or len(x_val) != len(val_df):
        raise ValueError("Représentations et DataFrames de longueurs différentes.")
    dossier = Path(output_dir)
    for sous_dossier in ("results", "predictions", "checkpoints"):
        (dossier / sous_dossier).mkdir(parents=True, exist_ok=True)
    y_train, y_val = labels_binaires(train_df).to_numpy(), labels_binaires(val_df).to_numpy()
    modele = LogisticRegression(class_weight="balanced", solver="liblinear",
                                max_iter=2000, random_state=graine)
    debut = time.perf_counter()
    modele.fit(x_train, y_train)
    temps = time.perf_counter() - debut
    scores = modele.predict_proba(x_val)[:, list(modele.classes_).index(1)]
    seuil = seuil_optimal_d(y_val, scores)
    nom = f"D1_AraBERT_gele_{representation}_LR"
    chemin = dossier / "checkpoints" / f"D1_{representation}_seed{graine}.joblib"
    joblib.dump({"classifieur": modele, "seuil": seuil,
        "representation": representation, "encodeur": NOM_ARABERT,
        "revision_encodeur": revision_encodeur, "max_length": 64,
        "pretraitement": "guillemets externes retires + ArabertPreprocessor"}, chemin)
    resultat = {"modele": nom, "graine": graine, "representation": representation,
        "pretraitement": "guillemets externes retires + ArabertPreprocessor",
        "seuil": seuil, "temps_entrainement_s": temps,
        "temps_extraction_commun_s": temps_extraction,
        "parametres_entraines": modele.coef_.size + modele.intercept_.size,
        "checkpoint": str(chemin), **metriques_d(y_val, scores, seuil)}
    ajouter_resultat_c(resultat, dossier / "results/results.csv")
    ajouter_resultat_c(resultat, dossier / "results/D1_comparaison.csv")
    pd.DataFrame({"id": val_df["id"].to_numpy(), "y_true": y_val,
        "y_pred": (scores >= seuil).astype(int), "prob_sarcastique": scores}).to_csv(
        dossier / "predictions" / f"D1_{representation}_val_seed{graine}.csv", index=False)
    return resultat


@torch.no_grad()
def predire_d2(modele, loader):
    appareil = next(modele.parameters()).device
    modele.eval()
    y, scores = [], []
    for etats, longueurs, labels in loader:
        logits = modele(etats.to(appareil), longueurs)
        scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        y.extend(labels.tolist())
    return np.asarray(y), np.asarray(scores)


def entrainer_tete_d2(etats_train, etats_val, train_df, val_df, dimension_entree,
                     graine=42, temps_extraction=0.0, revision_encodeur=None,
                     device=None, output_dir="experiences_src"):
    """Tête BiLSTM-attention D2, encodeur exclu de l'optimiseur.

    Fournir les états déjà extraits et stockés en float16 comme dans le notebook.
    Budget original : huit époques maximum, patience 2, Adam 0.001.
    """
    from torch.utils.data import DataLoader
    from .datasets_torch import DatasetD2, preparer_batch_d2
    from .models import TeteD2
    from .preprocess import labels_binaires
    from .config import NOM_ARABERT
    if len(etats_train) != len(train_df) or len(etats_val) != len(val_df):
        raise ValueError("États et DataFrames de longueurs différentes.")
    fixer_graine_c(graine)
    appareil = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dossier = Path(output_dir)
    for sous_dossier in ("results", "predictions", "checkpoints"):
        (dossier / sous_dossier).mkdir(parents=True, exist_ok=True)
    y_train, y_val = labels_binaires(train_df).to_numpy(), labels_binaires(val_df).to_numpy()
    train_loader = DataLoader(DatasetD2(etats_train, y_train), batch_size=32,
        shuffle=True, collate_fn=preparer_batch_d2,
        generator=torch.Generator().manual_seed(graine))
    val_loader = DataLoader(DatasetD2(etats_val, y_val), batch_size=32,
        shuffle=False, collate_fn=preparer_batch_d2)
    modele = TeteD2(dimension_entree).to(appareil)
    effectifs = np.bincount(y_train, minlength=2)
    if (effectifs == 0).any():
        raise ValueError("Train doit contenir les deux classes.")
    poids = torch.tensor(effectifs.sum() / (2 * effectifs), dtype=torch.float32, device=appareil)
    critere = nn.CrossEntropyLoss(weight=poids)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=0.001)
    meilleur_f1, meilleur_etat, meilleure_epoque, attente = -1.0, None, 0, 0
    historique = []
    if appareil.type == "cuda":
        torch.cuda.synchronize()
    debut = time.perf_counter()
    for epoque in range(1, 9):
        modele.train()
        somme, effectif = 0.0, 0
        for etats, longueurs, labels in train_loader:
            optimiseur.zero_grad(set_to_none=True)
            perte = critere(modele(etats.to(appareil), longueurs), labels.to(appareil))
            perte.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), max_norm=1.0)
            optimiseur.step()
            somme += perte.item() * len(labels)
            effectif += len(labels)
        y, scores = predire_d2(modele, val_loader)
        f1 = metriques_d(y, scores, .5)["val_f1_sarc"]
        historique.append({"epoque": epoque, "perte_train": somme / effectif,
                           "val_f1_seuil_05": f1})
        if f1 > meilleur_f1 + 1e-8:
            meilleur_f1, meilleure_epoque, attente = f1, epoque, 0
            meilleur_etat = {k: v.detach().cpu().clone() for k, v in modele.state_dict().items()}
        else:
            attente += 1
            if attente >= 2:
                break
    if appareil.type == "cuda":
        torch.cuda.synchronize()
    temps = time.perf_counter() - debut
    modele.load_state_dict(meilleur_etat)
    y, scores = predire_d2(modele, val_loader)
    seuil = seuil_optimal_d(y, scores)
    chemin = dossier / "checkpoints" / f"D2_tete_seed{graine}.pt"
    torch.save({"state_dict": meilleur_etat, "dimension_entree": dimension_entree,
        "hidden_size": 64, "dropout": 0.3, "encodeur": NOM_ARABERT,
        "revision_encodeur": revision_encodeur, "max_length": 64, "seuil": seuil,
        "pretraitement": "guillemets externes retires + ArabertPreprocessor"}, chemin)
    resultat = {"modele": "D2_AraBERT_gele_BiLSTM_attention", "graine": graine,
        "pretraitement": "guillemets externes retires + ArabertPreprocessor",
        "seuil": seuil, "temps_extraction_commun_s": temps_extraction,
        "temps_entrainement_s": temps,
        "parametres_entraines": sum(p.numel() for p in modele.parameters() if p.requires_grad),
        "meilleure_epoque": meilleure_epoque, "checkpoint": str(chemin),
        **metriques_d(y, scores, seuil)}
    ajouter_resultat_c(resultat, dossier / "results/results.csv")
    ajouter_resultat_c(resultat, dossier / "results/D2_resultat.csv")
    pd.DataFrame(historique).to_csv(dossier / "results" / f"D2_seed{graine}_historique.csv", index=False)
    pd.DataFrame({"id": val_df["id"].to_numpy(), "y_true": y,
        "y_pred": (scores >= seuil).astype(int), "prob_sarcastique": scores}).to_csv(
        dossier / "predictions" / f"D2_val_seed{graine}.csv", index=False)
    return resultat
