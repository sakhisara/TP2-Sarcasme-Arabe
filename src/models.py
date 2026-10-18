
import math
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .config import PAD_ID

class PoolingAttentionC(nn.Module):
    def __init__(self, dimension, dimension_attention=64):
        super().__init__()

        self.W = nn.Linear(
            dimension,
            dimension_attention
        )

        self.v = nn.Linear(
            dimension_attention,
            1,
            bias=False
        )

    def forward(self, etats, masque):
        # score_i = v^T tanh(W h_i)
        scores = self.v(
            torch.tanh(self.W(etats))
        ).squeeze(-1)

        # Le padding ne doit recevoir aucun poids.
        scores = scores.masked_fill(
            ~masque,
            float("-inf")
        )

        alpha = torch.softmax(scores, dim=1)

        # c = somme(alpha_i * h_i)
        contexte = torch.sum(
            etats * alpha.unsqueeze(-1),
            dim=1
        )

        return contexte, alpha

class PositionsSinusoidalesC(nn.Module):
    def __init__(self, dimension, max_len):
        super().__init__()

        positions = torch.arange(
            max_len
        ).float().unsqueeze(1)

        frequences = torch.exp(
            torch.arange(0, dimension, 2).float()
            * (-math.log(10000.0) / dimension)
        )

        encodage = torch.zeros(max_len, dimension)

        encodage[:, 0::2] = torch.sin(
            positions * frequences
        )
        encodage[:, 1::2] = torch.cos(
            positions * frequences
        )

        self.register_buffer(
            "encodage",
            encodage.unsqueeze(0)
        )

    def forward(self, x):
        return x + self.encodage[:, :x.size(1)]

class ModeleC(nn.Module):
    def __init__(self, matrice_embedding, config):
        super().__init__()

        self.config = dict(config)

        poids = torch.as_tensor(
            matrice_embedding,
            dtype=torch.float32
        ).clone()

        poids[PAD_ID].zero_()

        self.embedding = nn.Embedding.from_pretrained(
            poids,
            freeze=False,
            padding_idx=PAD_ID
        )

        dimension_embedding = poids.shape[1]
        self.dropout = nn.Dropout(config["dropout"])

        architecture = config["arch"]

        if architecture in {"C1", "C2"}:
            self.encodeur = nn.LSTM(
                input_size=dimension_embedding,
                hidden_size=config["hidden_size"],
                batch_first=True,
                bidirectional=True
            )

            dimension_sortie = 2 * config["hidden_size"]

            if architecture == "C2":
                self.attention = PoolingAttentionC(
                    dimension_sortie
                )

        elif architecture == "C3":
            dimension_sortie = 128

            self.projection = nn.Linear(
                dimension_embedding,
                dimension_sortie
            )

            self.positions = PositionsSinusoidalesC(
                dimension_sortie,
                config["max_len"]
            )

            couche = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=config["dropout"],
                batch_first=True
            )

            self.encodeur = nn.TransformerEncoder(
                couche,
                num_layers=2,
                enable_nested_tensor=False
            )

        elif architecture == "C4":
            self.convolutions = nn.ModuleList([
                nn.Conv1d(
                    dimension_embedding,
                    64,
                    kernel_size=k,
                    padding=k // 2
                )
                for k in [3, 5, 7]
            ])

            dimension_sortie = 64 * 3

        else:
            raise ValueError("Architecture inconnue.")

        self.classifieur = nn.Linear(
            dimension_sortie,
            2
        )

    def forward(
        self,
        ids,
        longueurs,
        retourner_attention=False
    ):
        masque = ids.ne(PAD_ID)
        x = self.dropout(self.embedding(ids))
        alpha = None

        if self.config["arch"] in {"C1", "C2"}:
            sequences = pack_padded_sequence(
                x,
                longueurs.cpu(),
                batch_first=True,
                enforce_sorted=False
            )

            etats_compacts, (h, _) = self.encodeur(
                sequences
            )

            if self.config["arch"] == "C1":
                # Derniers états avant et arrière hors padding.
                contexte = torch.cat(
                    [h[-2], h[-1]],
                    dim=-1
                )

            else:
                etats, _ = pad_packed_sequence(
                    etats_compacts,
                    batch_first=True,
                    total_length=ids.size(1)
                )

                contexte, alpha = self.attention(
                    etats,
                    masque
                )

        elif self.config["arch"] == "C3":
            x = self.projection(x) * math.sqrt(128)

            if self.config["positions"]:
                x = self.positions(x)

            etats = self.encodeur(
                x,
                src_key_padding_mask=~masque
            )

            # Moyenne uniquement sur les tokens réels.
            contexte = (
                etats * masque.unsqueeze(-1)
            ).sum(dim=1) / masque.sum(
                dim=1,
                keepdim=True
            )

        else:
            # C4 : CNN caractères + max-pooling masqué.
            sorties = []

            for convolution in self.convolutions:
                etats = torch.relu(
                    convolution(x.transpose(1, 2))
                )

                etats = etats.masked_fill(
                    ~masque.unsqueeze(1),
                    float("-inf")
                )

                sorties.append(etats.amax(dim=2))

            contexte = torch.cat(sorties, dim=1)

        logits = self.classifieur(
            self.dropout(contexte)
        )

        if retourner_attention:
            return logits, alpha

        return logits

class TeteD2(nn.Module):
    def __init__(
        self,
        dimension_entree,
        hidden_size=64,
        dropout=0.3
    ):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=dimension_entree,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.attention = PoolingAttentionC(
            dimension=2 * hidden_size
        )

        self.dropout = nn.Dropout(dropout)
        self.classifieur = nn.Linear(2 * hidden_size, 2)

    def forward(
        self,
        representations,
        longueurs,
        retourner_attention=False
    ):
        sequences = pack_padded_sequence(
            representations,
            longueurs.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        etats_compacts, _ = self.bilstm(sequences)

        etats, _ = pad_packed_sequence(
            etats_compacts,
            batch_first=True,
            total_length=representations.size(1)
        )

        positions = torch.arange(
            representations.size(1),
            device=representations.device
        ).unsqueeze(0)

        masque = positions < longueurs.to(
            representations.device
        ).unsqueeze(1)

        contexte, alpha = self.attention(
            etats,
            masque
        )

        logits = self.classifieur(
            self.dropout(contexte)
        )

        if retourner_attention:
            return logits, alpha

        return logits

class ModeleE(nn.Module):
    def __init__(self, nom_modele, multitache=False, dropout=0.1, revision=None):
        super().__init__()
        from transformers import AutoModel

        self.multitache = multitache

        self.enc = AutoModel.from_pretrained(
            nom_modele,
            revision=revision,
            add_pooling_layer=False
        )

        # Tous les paramètres de l'encodeur sont entraînables.
        for parametre in self.enc.parameters():
            parametre.requires_grad = True

        dimension = self.enc.config.hidden_size

        self.dropout = nn.Dropout(
            dropout
        )

        self.sarc = nn.Linear(dimension, 2)

        self.sent = (
            nn.Linear(dimension, 3)
            if multitache
            else None
        )

    def forward(self, **entrees):
        sortie = self.enc(**entrees)

        cls = self.dropout(
            sortie.last_hidden_state[:, 0]
        )

        logits_sarc = self.sarc(cls)

        logits_sent = (
            self.sent(cls)
            if self.sent is not None
            else None
        )

        return logits_sarc, logits_sent

def charger_modele_c(chemin):
    checkpoint = torch.load(
        chemin,
        map_location="cpu",
        weights_only=False
    )

    matrice_vide = np.zeros(
        checkpoint["embedding_shape"],
        dtype=np.float32
    )

    modele = ModeleC(
        matrice_vide,
        checkpoint["config"]
    )

    modele.load_state_dict(
        checkpoint["state_dict"]
    )

    return modele.eval(), checkpoint
