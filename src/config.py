"""Valeurs par défaut reprises du notebook, sans effet de bord."""
PAD_ID = 0
UNK_ID = 1
NOM_ARABERT = "aubmindlab/bert-base-arabertv02-twitter"
MODELES_E = {"AraBERT": NOM_ARABERT, "MARBERT": "UBC-NLP/MARBERTv2"}
SENTIMENTS_E = {"NEG": 0, "NEU": 1, "POS": 2}
CONFIG_C = {
    "seed": 42, "batch_size": 32, "max_epochs": 8, "patience": 2,
    "lr": 0.001, "dropout": 0.3, "hidden_size": 64,
    "freeze_epochs": 1, "arch": "C1", "embedding": "random",
    "positions": True, "methode": "standard"
}
# max_len est calculé sur TRAIN et doit être ajouté à CONFIG_C.
CONFIG_E = {
    "lr": 2e-5, "batch_size": 32, "micro_batch": 8,
    "max_epochs": 5, "patience": 2, "warmup_ratio": 0.10,
    "weight_decay": 0.01, "max_length": 64, "dropout": 0.1
}
