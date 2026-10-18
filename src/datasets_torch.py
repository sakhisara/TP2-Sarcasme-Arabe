"""Datasets et lots PyTorch extraits des parties C, D2 et E."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from .config import PAD_ID, UNK_ID, SENTIMENTS_E
from .preprocess import labels_binaires, textes_pour_e

def labels_c(df):
    return labels_binaires(df).to_numpy()

class TweetsDatasetC(Dataset):
    def __init__(self, df, max_len):
        self.sequences = [
            list(ids[:max_len]) or [UNK_ID]
            for ids in df["input_ids"]
        ]
        self.labels = labels_c(df)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.tensor(
                self.sequences[index],
                dtype=torch.long
            ),
            int(self.labels[index])
        )

def preparer_batch_c(batch):
    sequences, labels = zip(*batch)

    longueurs = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long
    )

    sequences_pad = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=PAD_ID
    )

    labels = torch.tensor(labels, dtype=torch.long)

    return sequences_pad, longueurs, labels

def creer_loaders_c(train_df, val_df, config):
    dataset_train = TweetsDatasetC(
        train_df, config["max_len"]
    )
    dataset_val = TweetsDatasetC(
        val_df, config["max_len"]
    )

    effectifs = np.bincount(
        dataset_train.labels,
        minlength=2
    )

    if (effectifs == 0).any():
        raise ValueError("Train doit contenir les deux classes.")

    generateur = torch.Generator().manual_seed(
        config["seed"]
    )

    sampler = None

    if config["methode"] == "oversampling":
        # Les poids viennent uniquement des labels de TRAIN.
        poids_exemples = 1.0 / effectifs[dataset_train.labels]

        sampler = WeightedRandomSampler(
            weights=torch.tensor(
                poids_exemples,
                dtype=torch.double
            ),
            num_samples=len(dataset_train),
            replacement=True,
            generator=generateur
        )

    train_loader = DataLoader(
        dataset_train,
        batch_size=config["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=preparer_batch_c,
        generator=generateur,
        num_workers=0
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=preparer_batch_c,
        num_workers=0
    )

    return train_loader, val_loader, effectifs

class DatasetD2(Dataset):
    def __init__(self, representations, labels):
        self.representations = representations
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.representations[index],
            int(self.labels[index])
        )

def preparer_batch_d2(batch):
    representations, labels = zip(*batch)

    longueurs = torch.tensor(
        [len(x) for x in representations],
        dtype=torch.long
    )

    representations_pad = pad_sequence(
        representations,
        batch_first=True,
        padding_value=0
    ).float()

    return (
        representations_pad,
        longueurs,
        torch.tensor(labels, dtype=torch.long)
    )

class DatasetE(Dataset):
    def __init__(self, df, textes):
        self.textes = textes
        self.sarcasme = labels_binaires(df).to_numpy()

        sentiment = (
            df["sentiment"].astype(str)
            .str.strip().str.upper()
            .map(SENTIMENTS_E)
        )

        if sentiment.isna().any():
            raise ValueError("Label de sentiment inconnu.")

        self.sentiment = sentiment.astype(int).to_numpy()

    def __len__(self):
        return len(self.textes)

    def __getitem__(self, index):
        return (
            self.textes[index],
            int(self.sarcasme[index]),
            int(self.sentiment[index])
        )

def creer_loaders_e(nom_court, tokenizer, graine, train_df, val_df, config):
    dataset_train = DatasetE(
        train_df,
        textes_pour_e(train_df, nom_court)
    )

    dataset_val = DatasetE(
        val_df,
        textes_pour_e(val_df, nom_court)
    )

    def preparer_batch(batch):
        textes, sarcasmes, sentiments = zip(*batch)

        entrees = tokenizer(
            list(textes),
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt"
        )

        return (
            entrees,
            torch.tensor(sarcasmes, dtype=torch.long),
            torch.tensor(sentiments, dtype=torch.long)
        )

    train_loader = DataLoader(
        dataset_train,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=preparer_batch,
        generator=torch.Generator().manual_seed(graine)
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=preparer_batch
    )

    return train_loader, val_loader, dataset_train.sarcasme
