import numpy as np
import torch
from tqdm.auto import tqdm

@torch.no_grad()
def extraire_representations_d(textes, encodeur_d, tokenizer_d, device="cpu", batch_size=32, max_length=64):
    DEVICE_D = torch.device(device)
    BATCH_D, MAX_LEN_D = batch_size, max_length
    encodeur_d.to(DEVICE_D).eval()
    for parametre in encodeur_d.parameters():
        parametre.requires_grad_(False)
    vecteurs_cls = []
    vecteurs_moyens = []

    for debut in tqdm(
        range(0, len(textes), BATCH_D),
        desc="Extraction AraBERT"
    ):
        lot = textes[debut:debut + BATCH_D]

        entrees = tokenizer_d(
            lot,
            padding=True,
            truncation=True,
            max_length=MAX_LEN_D,
            return_tensors="pt"
        )

        entrees = {
            cle: valeur.to(DEVICE_D)
            for cle, valeur in entrees.items()
        }

        sorties = encodeur_d(**entrees)

        # Dimensions : batch × tokens × dimension cachée.
        etats = sorties.last_hidden_state

        # Premier token : CLS.
        cls = etats[:, 0, :]

        # Masque : 1 pour token réel, 0 pour padding.
        masque = entrees["attention_mask"].unsqueeze(-1)
        masque = masque.to(etats.dtype)

        moyenne = (
            (etats * masque).sum(dim=1)
            / masque.sum(dim=1).clamp(min=1)
        )

        vecteurs_cls.append(cls.cpu().numpy())
        vecteurs_moyens.append(moyenne.cpu().numpy())

    return (
        np.concatenate(vecteurs_cls),
        np.concatenate(vecteurs_moyens)
    )

@torch.no_grad()
def extraire_tokens_d2(textes, encodeur_d, tokenizer_d, device="cpu", batch_size=32, max_length=64):
    DEVICE_D = torch.device(device)
    BATCH_D, MAX_LEN_D = batch_size, max_length
    encodeur_d.to(DEVICE_D).eval()
    for parametre in encodeur_d.parameters():
        parametre.requires_grad_(False)
    encodeur_d.to(DEVICE_D)
    encodeur_d.eval()

    representations = []
    identifiants = []

    for debut in tqdm(
        range(0, len(textes), BATCH_D),
        desc="Extraction des tokens D2"
    ):
        entrees = tokenizer_d(
            textes[debut:debut + BATCH_D],
            padding=True,
            truncation=True,
            max_length=MAX_LEN_D,
            return_tensors="pt"
        )

        longueurs = entrees["attention_mask"].sum(dim=1)
        ids_cpu = entrees["input_ids"]

        entrees_gpu = {
            cle: valeur.to(DEVICE_D)
            for cle, valeur in entrees.items()
        }

        etats = encodeur_d(
            **entrees_gpu
        ).last_hidden_state

        # Stockage CPU en float16 pour réduire la mémoire.
        etats_cpu = etats.to(
            device="cpu",
            dtype=torch.float16
        )

        for i, longueur in enumerate(longueurs.tolist()):
            representations.append(
                etats_cpu[i, :longueur].clone()
            )
            identifiants.append(
                ids_cpu[i, :longueur].clone()
            )

    return representations, identifiants

