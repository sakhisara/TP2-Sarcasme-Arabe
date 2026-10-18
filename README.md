# TP2 — Détection du sarcasme en arabe
Ce projet compare plusieurs méthodes de détection du sarcasme sur les tweets
arabes du corpus **ArSarcasm-v2**. Le travail va d'une baseline TF-IDF à des modèles
pré-entraînés utilisés directement, gelés ou fine-tunés. Il étudie aussi le
déséquilibre des classes, l'apport du sentiment et la généralisation à d'autres
types de tweets.

Le notebook présente les expériences et leurs sorties. Le compte rendu développe
les réponses aux questions du TP. Les fichiers de `src/` regroupent les fonctions
réutilisables, et les CSV permettent de vérifier les scores sans réentraîner les
modèles.

## Organisation du dépôt

```text
TP2-Sarcasme-Arabe/
├── README.md
├── requirements.txt
├── notebooks/
│   └── TP2_Sarcasme_Arabe.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocess.py
│   ├── datasets_torch.py
│   ├── models.py
│   ├── representations.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
├── tests/
├── results/
│   ├── results.csv
│   ├── F/
│   └── G/
├── predictions/
└── rapport/
    └── TP2_VersionB_Sarcasme_SARA_SAKHI.pdf
```

Les fonctions de chargement sont dans `data.py`, les nettoyages dans
`preprocess.py`, les architectures dans `models.py`, les entraînements dans
`train.py` et les métriques dans `evaluate.py`. Les autres modules contiennent
les configurations, les lots PyTorch, l'extraction des représentations et le
chargement des checkpoints.

Le notebook conserve les boucles de comparaison des expériences A à G. Les
modules Python l'accompagnent ; ils ne lancent pas automatiquement toutes les
expériences à l'import.

## Environnement

Les expériences ont été réalisées dans **Google Colab**, avec :

- Python **3.13.15** ;
- PyTorch **2.11.0+cu130**, selon la vérification enregistrée dans le notebook ;
- GPU **Tesla T4** pour les réseaux de neurones ;
- bibliothèques principales : torch, transformers, datasets, evaluate,
  scikit-learn, gensim, pyarabic, emoji, arabert, pandas, NumPy, SciPy,
  matplotlib, joblib, requests et tqdm.

Les versions de l'environnement des expériences sont conservées dans
`requirements.txt`. Depuis la racine du projet :

```bash
python -m pip install -r requirements.txt
```

Ce fichier correspond à l'environnement Colab utilisé. Les variantes de PyTorch
et de CUDA peuvent demander une adaptation sur un autre système. Les durées et
la reproductibilité numérique dépendent aussi du matériel et des versions.

## Données et protocole

Source : [dépôt officiel ArSarcasm-v2](https://github.com/iabufarha/ArSarcasm-v2).
Les fichiers sont téléchargés dans le notebook depuis le dossier `ArSarcasm-v2/`
du dépôt :

- `training_data.csv` : **12 548 tweets** ;
- `testing_data.csv` : **3 000 tweets**, correspondant au test officiel.

La validation représente **10 % du fichier d'entraînement**, avec
`random_state=42` et une stratification sur `sarcasm`. Avant nettoyage, le split
donne **11 293 tweets dans train** et **1 255 dans val**.

Le nettoyage de train retire, dans cet ordre, **2 tweets présents dans test**,
**96 lignes appartenant à des groupes aux labels contradictoires**, puis
**282 répétitions restantes**. Le train final contient **10 913 tweets**.
Validation et test conservent leurs effectifs.

Les labels de sarcasme sont convertis en **0 = non sarcastique** et
**1 = sarcastique**. Le texte est l'entrée des modèles. Le sentiment est une
cible auxiliaire dans E2, et le dialecte sert aux analyses par sous-groupe ;
ces annotations ne sont pas ajoutées au texte en entrée.

Le vocabulaire, les poids TF-IDF, les poids de classes et le ré-échantillonnage
sont construits sur train. Les décisions de sélection des modèles et des seuils
s'appuient sur validation. Les évaluations finales et les analyses de la partie F
utilisent le test officiel. Les graines des modèles finaux sont **42, 43 et 44**.

La métrique principale est le **F1 de la classe sarcastique**, accompagné du
macro-F1 et de la PR-AUC, calculée dans le notebook par intégration de la courbe
précision-rappel. L'accuracy sert notamment à illustrer les limites du
classifieur « jamais sarcastique » ; elle ne sert pas à choisir le meilleur modèle.

## Travail réalisé

### Audit et prétraitement

L'audit porte sur les tweets bruts, les URL, mentions, hashtags, emojis, rires,
ponctuations répétées, diacritiques, tatweel, lettres répétées, caractères latins
et chiffres arabes orientaux. Les guillemets externes ont été étudiés comme
artefact de format candidat, avec leur fréquence et leur association au label.
Les doublons exacts et après normalisation ont également été examinés.

Le prétraitement dépend de la famille de modèles :

- **TF-IDF et B1/B2** : retrait des guillemets externes et harmonisation des espaces.
- **Réseaux C** : même nettoyage, retrait des diacritiques et du tatweel,
  normalisation des variantes d'alif, puis découpage en mots et symboles.
  Le vocabulaire est construit sur train avec une fréquence minimale de 2.
  Les tokens absents sont remplacés par `<UNK>`, et le taux d'OOV est mesuré.
- **AraVec** : application de sa fonction officielle `clean_str` aux clés
  recherchées dans le vocabulaire pré-entraîné.
- **AraBERT-twitter** : retrait des guillemets externes et utilisation de
  `ArabertPreprocessor`.
- **MARBERTv2** : retrait des guillemets externes et harmonisation des espaces.

Les modèles des parties D et E utilisent leur tokenizer associé, une longueur maximale 
de 64 tokens et un padding dynamique par lot.
Les emojis, rires et ponctuations sont conservés dans le nettoyage léger; 
leur suppression conjointe est étudiée séparément en F4.

### A — Exploration et baseline

Calcul des taux de sarcasme par ensemble, dialecte et sentiment, évaluation du
classifieur « jamais sarcastique », puis entraînement de **TF-IDF mots 1–2 et
caractères 2–5 + régression logistique `class_weight="balanced"`**. Le seuil est
choisi sur validation. Des règles utilisant uniquement une URL, une mention ou
les guillemets externes servent à examiner les raccourcis du corpus.

### B — Modèles utilisés sans nouvel entraînement

- **B1** : zero-shot NLI avec
  `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, en testant des
  hypothèses en arabe, français et anglais. Les tweets restent en arabe.
- **B2** : modèle de sentiment
  `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`, suivi de la règle
  « sentiment négatif prédit ⇒ sarcastique ».

La meilleure formulation B1 sur validation est le français (**F1 = 0,4276**).
B2 obtient **0,4688**. Le chevauchement possible entre les corpus de fine-tuning
du modèle de sentiment et les sources d'ArSarcasm-v2 limite l'interprétation de B2.

### C — Embeddings statiques et réseaux PyTorch

Comparaison de **C1 BiLSTM**, **C2 BiLSTM avec pooling attentionnel implémenté**,
**C3 Transformer à deux couches, dimension 128 et quatre têtes**, avec et sans
encodage positionnel. Les embeddings sont aléatoires ou issus d'**AraVec Twitter
unigrammes 100 dimensions**, figés ou figés puis dégelés.

Le **char-CNN C4 optionnel a aussi été exécuté** : son F1 de validation est
**0,3427** dans cette exécution. Il ne fait pas partie des modèles finaux retenus.
Parmi C1–C3, C1 avec AraVec dégelés obtient **0,4930** sur validation avant la
comparaison des méthodes de déséquilibre. La perte pondérée atteint ensuite
**0,5362**, contre **0,5079** pour le sur-échantillonnage et **0,5122** pour la
perte standard avec seuil optimisé. Une visualisation de l'attention de C2 complète
la comparaison des scores.

### D — Encodeur gelé

Avec `aubmindlab/bert-base-arabertv02-twitter` gelé :

- **D1** compare `[CLS]` et la moyenne masquée des tokens, suivis d'une régression
  logistique : F1 de validation de **0,6062** et **0,6247** ;
- **D2** utilise les états de tous les tokens avec une tête BiLSTM-attention :
  F1 de validation de **0,6538**. L'attention est visualisée sur cinq tweets
  sarcastiques.

### E — Fine-tuning mono-tâche et multi-tâche

Comparaison d'**AraBERT-twitter** et de **`UBC-NLP/MARBERTv2`** : E1 apprend le
sarcasme, et E2 partage l'encodeur entre deux têtes, sarcasme et sentiment.
La perte E2 est `L_sarc + lambda * L_sent`, avec **lambda = 0,3 ; 0,5 ; 1**.
La prédiction de sentiment n'est pas donnée directement à la tête sarcasme.

Réglages utilisés : learning rate **2e-5**, batch effectif **32** avec micro-batch
**8**, maximum **5 époques**, patience **2**, warmup **10 %**, weight decay
**0,01** et FP16 sur GPU. La perte sarcasme est pondérée. Le seuil et l'early
stopping utilisent le F1 sarcastique de validation.

AraBERT-twitter est retenu pour les modèles finaux E1 et E2, avec **lambda = 0,3
pour E2**. Sur trois graines, leurs F1 de validation sont respectivement
**0,6865 ± 0,0009** et **0,6946 ± 0,0018**.

### F — Généralisation et analyse des erreurs

- **F1** : comparaison des proportions de sarcasme et des seuils ; courbe du F1
  selon le seuil sur test, uniquement pour une analyse a posteriori.
- **F2** : retrait des tweets `egypt` de train, réentraînement du meilleur E et
  évaluation sur les 306 tweets égyptiens du test. Le F1 passe de **0,7829 à
  0,7771**, avec un seuil choisi sur validation pour chaque modèle.
- **F3** : évaluation avec/sans URL et réentraînement sans URL ni mentions.
- **F4** : suppression conjointe des emojis, rires et ponctuations répétées sur
  les vrais positifs initiaux, en conservant le modèle et le seuil.
- **F5** : résultats sur trois graines, bootstrap à 95 % avec **1 000
  rééchantillonnages**, et test exact de McNemar E1/E2 sur la graine 42
  (**p = 0,8072**, pas de différence significative des taux d'erreur à 5 %).
- **F6** : analyse de **30 erreurs d'E2**, 15 faux négatifs et 15 faux positifs,
  avec une catégorie et une justification pour chaque tweet. Le croisement avec
  les textes aux labels contradictoires a été ajouté sans réentraîner les modèles.

Les catégories F6 sont : contexte (9), bruit d'annotation suspecté (8), ironie sans
marqueur (5), marqueur mal interprété (5) et dialecte (3). Aucun des 30 tweets ne
correspond aux groupes contradictoires retrouvés dans l'audit train avant
nettoyage + test. Les huit labels suspects ne sont donc pas confirmés comme faux.

### G — Comparaison finale

Comparaison de la qualité, de la robustesse, du temps d'entraînement, de la taille
des modèles et de leur latence CPU. La matrice de confusion est présentée pour
E2, graine 42, avec son seuil de validation.

## Résultats principaux sur le test officiel

Les moyennes et écarts-types ci-dessous portent sur les graines **42, 43 et 44**.
La latence est la moyenne mesurée sur **30 tweets**, avec **deux threads CPU**.

| Modèle final | F1 sarcastique moyen ± écart-type | F1 graine 42 | Latence CPU (ms/tweet) | Taille (Mio) |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + régression logistique | 0,5424 ± 0,0000 | 0,5424 | 1,80 | 27,60 |
| C — BiLSTM AraVec dégelés, perte pondérée | 0,5468 ± 0,0077 | 0,5499 | 2,63 | 6,25 |
| D — AraBERT gelé + BiLSTM-attention | 0,5710 ± 0,0110 | 0,5625 | 116,45 | 517,39 |
| E1 — AraBERT-twitter mono-tâche | 0,6000 ± 0,0080 | 0,5908 | 108,03 | 513,48 |
| E2 — AraBERT-twitter multi-tâche | **0,6094 ± 0,0094** | **0,5998** | 107,27 | 513,49 |

E2 est retenu pour un **déploiement pilote**. Pour la graine 42, son seuil est
**0,8317**, choisi sur validation. L'entraînement moyen des trois exécutions E2
prend **291,45 secondes**. La matrice de confusion sur test contient **1 863 vrais
négatifs, 316 faux positifs, 334 faux négatifs et 487 vrais positifs**.

Le seuil optimal calculé après coup sur test (**0,6976 pour E2**) sert uniquement
à l'analyse F1. Le F1 de **0,6386** associé à ce seuil ne remplace pas le score
final de **0,5998**, obtenu avec le seuil de validation.

## Exécution dans Colab

1. Ouvrir `notebooks/TP2_Sarcasme_Arabe.ipynb` dans Google Colab.
2. Activer une T4 pour les cellules d'extraction et d'entraînement des réseaux.
3. Installer les dépendances prévues par le notebook. Pour employer le fichier
   figé dans Colab, le rendre accessible à la session puis exécuter :

   ```python
   %pip install -r /content/TP2-Sarcasme-Arabe/requirements.txt
   ```

   Adapter ce chemin à l'endroit où le dépôt a été importé ou cloné.

4. Pour reproduire les expériences, suivre les cellules dans l'ordre :
   environnement, données, audit, puis parties A à G. Les cellules interactives
   d'import et d'annotation nécessitent les fichiers demandés.
5. Sauvegarder régulièrement les résultats et le notebook hors de la session
   Colab. Les fichiers locaux de la session ne constituent pas une sauvegarde.

Pour consulter le travail déjà réalisé, les sorties du notebook, les CSV et le
rapport suffisent. Une nouvelle exécution complète relance les entraînements et
peut réécrire des fichiers portant les mêmes noms. En particulier, la génération
initiale des erreurs F6 recrée des catégories vides : conserver les CSV annotés
avant de relancer cette cellule.

## Tests et vérification des prédictions

Depuis la racine du dépôt, lancer :

```bash
python -m unittest discover -s tests -v
```

Les tests utilisent de petits exemples artificiels pour vérifier les labels,
la conservation de marqueurs dans le nettoyage léger, le vocabulaire construit
sur train, le split, le nettoyage des doublons, les métriques, l'alignement des
prédictions et la conservation des annotations F6. Deux tests supplémentaires
vérifient les sorties des architectures C et le masquage de l'attention.
Ils sont ignorés si PyTorch n'est pas installé. Aucun entraînement ni
téléchargement de modèle n'est nécessaire pour ces tests.

L'extraction des modules a été vérifiée avec **14 tests réussis** ; les deux tests
PyTorch n'ont pas été exécutés dans l'environnement de préparation des sources.
Les entraînements et le chargement des checkpoints n'y ont pas été relancés.

Les CSV du test suivent le nom `predictions/<modele>_seed<graine>.csv`, par exemple
`E2_seed42.csv`. Ils contiennent `id`, `y_true`, `y_pred` et une colonne de
probabilité de sarcasme. Chaque graine conserve son propre seuil de validation.

Pour recalculer les métriques d'un fichier existant :

```bash
python -m src.evaluate predictions/E2_seed42.csv
```

Pour comparer E1 et E2 avec McNemar :

```bash
python -m src.evaluate predictions/E1_seed42.csv --comparer predictions/E2_seed42.csv
```

Ces commandes lisent les prédictions sauvegardées sans entraîner de modèle.

## Où retrouver les résultats

| Contenu | Emplacement |
| --- | --- |
| Journal des expériences | `results/results.csv` |
| Taux de sarcasme et audit | `results/taux_sarcasme.csv`, fichiers `audit_*.csv` et `bilan_nettoyage.csv` |
| Comparaison C1/C2/C3 | `results/C_comparaison_architectures.csv` |
| Méthodes de déséquilibre | `results/C_Q6_desequilibre.csv` |
| Comparaisons D et E | `results/D1_comparaison.csv`, `D2_resultat.csv`, `E1_comparaison.csv`, `E2_comparaison_lambda.csv` |
| Scores et seuils F1 | `results/F/F1_scores_test.csv`, `F1_analyse_seuils.csv`, `F1_courbe_seuil_test.png` |
| Dialecte non vu | `results/F/F2_dialecte_non_vu.csv` |
| Raccourcis et suppression de marqueurs | `results/F/F3_raccourcis.csv`, `F4_suppression_marqueurs.csv` |
| Bootstrap et comparaison des graines | `results/F/F5_bootstrap_par_graine.csv`, `F5_moyenne_ecart_type.csv` |
| McNemar | `results/F/F5_McNemar.csv` |
| Erreurs E2 annotées et vérifiées | `results/F/F6_erreurs_E2_complete.csv` |
| Détail des contradictions de l'audit étendu | `results/F/F6_doublons_contradictoires_train_test.csv` |
| Comparaison finale et matrice de confusion | `results/G/G_comparaison_finale.csv`, `G_matrice_confusion.png` |
| Prédictions finales | `predictions/` |
| Compte rendu | `rapport/TP2_VersionB_Sarcasme_SARA_SAKHI.pdf` |

Dans cette table, les noms abrégés d'une même ligne se trouvent dans le même
dossier que le premier fichier indiqué. Les autres sorties détaillées et les
historiques d'entraînement restent dans `results/`.

## Ressources utilisées

Les expériences utilisent AraBERT-twitter, MARBERTv2 et les modèles
pré-entraînés de la partie B, chargés depuis Hugging Face dans le notebook.

Pour la partie C, nous avons utilisé les embeddings AraVec Twitter
unigrammes de 100 dimensions et leur fonction de nettoyage `clean_str`.

Les checkpoints ont été sauvegardés et conservés dans une sauvegarde
personnelle. Ils ne sont pas inclus dans le dépôt en raison de leur
taille importante et parce qu’ils ne sont pas explicitement demandés
dans la liste des livrables du TP.

Le notebook, les résultats dans `results/` et les prédictions dans
`predictions/` permettent de consulter les expériences réalisées et
de recalculer les scores à partir des prédictions sauvegardées.

## Limites et suites possibles

Le taux de sarcasme est de **17,29 % sur validation**, contre **27,37 % sur test**.
La distribution des données n'est donc pas identique. Le modèle E2 est aussi
fragile sur les tweets avec URL : **F1 = 0,0833**, contre **0,6075 sans URL**, pour
la graine 42. Ce sous-groupe ne comporte que **16 tweets sarcastiques sur 440** ;
les résultats doivent être interprétés en tenant compte de cet effectif.

Le corpus ne contient que **35 tweets `magreb` dans train après nettoyage et 2
dans test**, tous deux non sarcastiques. Il ne permet pas de valider un usage
sur la darija marocaine. Enfin, les 30 erreurs relues et le benchmark de 30 tweets
donnent une vue limitée des erreurs et de la latence.

Les suites possibles sont d'enrichir les données dialectales, de faire relire
les labels ambigus, de vérifier la calibration sur des données récentes, et de
tester davantage les changements de domaine. Une réduction de la taille du
modèle pourrait aussi être étudiée, en contrôlant son effet sur le F1. Ces pistes
ne correspondent pas à des améliorations déjà mesurées dans ce TP.
