import gc
import re
import joblib
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .models import charger_modele_c, TeteD2
from .preprocess import (nettoyer_baseline, normaliser_comparaison,
                         retirer_guillemets_f, retirer_raccourcis_f)

class PredicteurF:
    def __init__(self, spec, device="cpu"):
        from transformers import AutoModel, AutoTokenizer
        from arabert.preprocess import ArabertPreprocessor
        self.device = torch.device(device)
        self.spec = spec
        self.type = spec["type"]

        if self.type == "baseline":
            self.sauvegarde = joblib.load(
                spec["checkpoint"]
            )
            self.seuil = self.sauvegarde["seuil"]

        elif self.type == "C":
            self.modele, self.sauvegarde = charger_modele_c(
                spec["checkpoint"]
            )
            self.modele.to(self.device).eval()

            self.seuil = self.sauvegarde["seuil"]
            self.vocabulaire = self.sauvegarde["vocabulaire"]
            self.max_length = self.sauvegarde["config"]["max_len"]

        elif self.type == "D2":
            self.sauvegarde = torch.load(
                spec["checkpoint"],
                map_location="cpu",
                weights_only=False
            )

            self.seuil = self.sauvegarde["seuil"]
            self.max_length = self.sauvegarde["max_length"]

            nom = self.sauvegarde["encodeur"]
            revision = self.sauvegarde.get("revision_encodeur")

            self.tokenizer = AutoTokenizer.from_pretrained(
                nom,
                revision=revision
            )

            self.encodeur = AutoModel.from_pretrained(
                nom,
                revision=revision
            ).to(self.device).eval()

            self.preprocesseur = ArabertPreprocessor(
                model_name=nom
            )

            self.modele = TeteD2(
                dimension_entree=self.sauvegarde["dimension_entree"],
                hidden_size=self.sauvegarde["hidden_size"],
                dropout=self.sauvegarde["dropout"]
            ).to(self.device)

            self.modele.load_state_dict(
                self.sauvegarde["state_dict"]
            )
            self.modele.eval()

        elif self.type == "E":
            self.sauvegarde = torch.load(
                spec["checkpoint"],
                map_location="cpu",
                weights_only=False
            )

            nom = self.sauvegarde["nom_modele"]
            revision = self.sauvegarde.get("revision_encodeur")

            self.seuil = self.sauvegarde["seuil"]
            self.max_length = self.sauvegarde["config"]["max_length"]

            self.tokenizer = AutoTokenizer.from_pretrained(
                nom,
                revision=revision
            )

            # Même architecture que ModeleE, avec les noms
            # d'attributs correspondant au checkpoint.
            class ReseauEF(nn.Module):
                def __init__(self, sauvegarde):
                    super().__init__()

                    self.enc = AutoModel.from_pretrained(
                        sauvegarde["nom_modele"],
                        revision=sauvegarde.get("revision_encodeur"),
                        add_pooling_layer=False
                    )

                    h = self.enc.config.hidden_size

                    self.dropout = nn.Dropout(
                        sauvegarde["config"]["dropout"]
                    )
                    self.sarc = nn.Linear(h, 2)

                    self.sent = (
                        nn.Linear(h, 3)
                        if sauvegarde["multitache"]
                        else None
                    )

                def forward(self, **entrees):
                    cls = self.dropout(
                        self.enc(**entrees).last_hidden_state[:, 0]
                    )
                    return self.sarc(cls)

            self.modele = ReseauEF(
                self.sauvegarde
            ).to(self.device)

            self.modele.load_state_dict(
                self.sauvegarde["state_dict"]
            )
            self.modele.eval()

            self.preprocesseur = (
                ArabertPreprocessor(model_name=nom)
                if "arabert" in nom.lower()
                else None
            )

        else:
            raise ValueError("Type de modèle inconnu.")

    def preparer_plm(self, texte):
        texte = retirer_guillemets_f(texte)

        if self.preprocesseur is not None:
            return self.preprocesseur.preprocess(texte)

        return " ".join(texte.split())

    @torch.no_grad()
    def scores(self, textes):
        textes = list(map(str, textes))

        if self.spec.get("sans_raccourcis", False):
            textes = [
                retirer_raccourcis_f(t)
                for t in textes
            ]

        if not textes:
            return np.array([], dtype=float)

        if self.type == "baseline":
            X = self.sauvegarde["tfidf"].transform(
                [nettoyer_baseline(t) for t in textes]
            )

            classifieur = self.sauvegarde["classifieur"]
            indice = list(classifieur.classes_).index(1)

            return classifieur.predict_proba(X)[:, indice]

        resultats = []

        for debut in range(0, len(textes), 16):
            lot = textes[debut:debut + 16]

            if self.type == "C":
                sequences = []

                for texte in lot:
                    tokens = (
                        list(normaliser_comparaison(texte))
                        if self.sauvegarde["config"]["arch"] == "C4"
                        else re.findall(r"\w+|[^\w\s]", normaliser_comparaison(texte))
                    )

                    ids = [
                        self.vocabulaire.get(token, 1)
                        for token in tokens
                    ][:self.max_length] or [1]

                    sequences.append(
                        torch.tensor(ids, dtype=torch.long)
                    )

                longueurs = torch.tensor(
                    [len(x) for x in sequences]
                )

                ids = pad_sequence(
                    sequences,
                    batch_first=True,
                    padding_value=0
                ).to(self.device)

                logits = self.modele(ids, longueurs)

            else:
                entrees = self.tokenizer(
                    [self.preparer_plm(t) for t in lot],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                longueurs = entrees["attention_mask"].sum(1)

                entrees = {
                    cle: valeur.to(self.device)
                    for cle, valeur in entrees.items()
                }

                if self.type == "D2":
                    etats = self.encodeur(
                        **entrees
                    ).last_hidden_state

                    # Même précision des représentations
                    # que le cache utilisé pour entraîner D2.
                    etats = etats.half().float()

                    logits = self.modele(
                        etats,
                        longueurs
                    )

                else:
                    logits = self.modele(**entrees)

            resultats.extend(
                torch.softmax(
                    logits.float(),
                    dim=-1
                )[:, 1].cpu().tolist()
            )

        return np.array(resultats)

    def fermer(self):
        for attribut in [
            "modele", "encodeur",
            "sauvegarde", "tokenizer"
        ]:
            if hasattr(self, attribut):
                delattr(self, attribut)

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
