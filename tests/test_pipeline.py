import unittest
import numpy as np
import pandas as pd
from src.preprocess import (labels_binaires, nettoyer_baseline, normaliser_comparaison,
                            tokeniser_c, construire_vocabulaire, tokens_vers_ids, taux_oov)
from src.data import separer_training, nettoyer_train, preparer_donnees_c
from src.evaluate import (mesurer_c, choisir_seuil_c, mcnemar_predictions,
                          croiser_erreurs_conflits)


class TestPretraitement(unittest.TestCase):
    def test_labels(self):
        df = pd.DataFrame({'sarcasm': ['TRUE', 'false', True, False, 1, 0]})
        self.assertEqual(labels_binaires(df).tolist(), [1, 0, 1, 0, 1, 0])

    def test_label_inconnu_refuse(self):
        for valeur in ['inconnu', None, 2]:
            with self.subTest(valeur=valeur), self.assertRaises(ValueError):
                labels_binaires(pd.DataFrame({'sarcasm': [valeur]}))

    def test_normalisation_et_marqueurs(self):
        self.assertEqual(normaliser_comparaison(' "أَهْـلًا   ههههه 😊!!!" '),
                         'اهلا ههههه 😊!!!')

    def test_nettoyage_baseline_conserve_indices(self):
        self.assertEqual(nettoyer_baseline(' "  ههههه   احبييي 😊!!!  " '),
                         'ههههه احبييي 😊!!!')

    def test_tokenisation_conserve_symboles(self):
        tokens = tokeniser_c('ههههه 😊!!!')
        self.assertIn('ههههه', tokens)
        self.assertIn('😊', tokens)
        self.assertEqual(tokens.count('!'), 3)

    def test_frequence_et_inconnus(self):
        vocab = construire_vocabulaire([['commun', 'rare'], ['commun']], 2)
        self.assertIn('commun', vocab)
        self.assertNotIn('rare', vocab)
        self.assertEqual(tokens_vers_ids(['commun', 'absent'], vocab), [vocab['commun'], 1])
        self.assertEqual(taux_oov([['commun', 'absent']], vocab)['taux_OOV_pct'], 50)
        self.assertEqual(taux_oov([], vocab)['taux_OOV_pct'], 0)


class TestDonnees(unittest.TestCase):
    def test_split_impose_reproductible(self):
        full = pd.DataFrame({'id': range(120), 'sarcasm': [False]*100 + [True]*20})
        a, b = separer_training(full)
        c, d = separer_training(full)
        self.assertEqual((len(a), len(b)), (108, 12))
        self.assertTrue(set(a.id).isdisjoint(b.id))
        self.assertEqual(b.sarcasm.sum(), 2)
        self.assertTrue(a.equals(c) and b.equals(d))

    def test_nettoyage_chevauchements_contradictions_repetitions(self):
        train = pd.DataFrame({'id': range(7),
            'tweet': ['commun_test', 'contradiction', 'contradiction',
                      'أهلا', 'اهلا', 'unique', 'autre'],
            'sarcasm': [0, 0, 1, 1, 1, 0, 1]})
        test = pd.DataFrame({'id': [0], 'tweet': ['commun_test'], 'sarcasm': [1]})
        original_train, original_test = train.copy(deep=True), test.copy(deep=True)
        propre, bilan, conflits = nettoyer_train(train, test)
        self.assertEqual(bilan['retires_chevauchement_test'], 1)
        self.assertEqual(bilan['retires_contradictions_train'], 2)
        self.assertEqual(bilan['retires_repetitions_train'], 1)
        self.assertEqual(len(propre), 3)
        self.assertEqual(len(conflits), 2)
        self.assertEqual(list(propre.columns), list(train.columns))
        self.assertTrue(train.equals(original_train) and test.equals(original_test))
        cles_test = set(test.tweet.map(normaliser_comparaison))
        self.assertTrue(set(propre.tweet.map(normaliser_comparaison)).isdisjoint(cles_test))

    def test_vocabulaire_appris_uniquement_sur_train(self):
        train = pd.DataFrame({'tweet': ['موجود موجود'], 'sarcasm': [0]})
        val = pd.DataFrame({'tweet': ['غريب غريب'], 'sarcasm': [1]})
        test = pd.DataFrame({'tweet': ['مختلف مختلف'], 'sarcasm': [0]})
        _, v, t, vocab, longueur = preparer_donnees_c(train, val, test)
        self.assertNotIn('غريب', vocab)
        self.assertNotIn('مختلف', vocab)
        self.assertEqual(v.input_ids.iloc[0], [1, 1])
        self.assertEqual(t.input_ids.iloc[0], [1, 1])
        self.assertEqual(longueur, 2)


class TestEvaluation(unittest.TestCase):
    def test_jamais_sarcastique(self):
        resultat = mesurer_c(np.array([0, 0, 0, 1]), np.zeros(4), .5)
        self.assertEqual(resultat['val_accuracy'], .75)
        self.assertEqual(resultat['val_f1_sarc'], 0)

    def test_seuil_optimise_sans_entrainement(self):
        y = np.array([0, 1, 1, 0])
        scores = np.array([.1, .4, .8, .3])
        seuil = choisir_seuil_c(y, scores)
        self.assertAlmostEqual(seuil, .4)
        self.assertEqual(mesurer_c(y, scores, seuil)['val_f1_sarc'], 1)

    def test_mcnemar_aligne_par_id(self):
        a = pd.DataFrame({'id': [2, 1, 3], 'y_true': [0, 1, 1], 'y_pred': [0, 0, 1]})
        b = pd.DataFrame({'id': [1, 3, 2], 'y_true': [1, 1, 0], 'y_pred': [1, 0, 0]})
        r = mcnemar_predictions(a, b)
        self.assertEqual(r['A_correct_B_incorrect'], 1)
        self.assertEqual(r['A_incorrect_B_correct'], 1)
        self.assertEqual(r['p_value'], 1)

    def test_mcnemar_refuse_labels_incompatibles(self):
        a = pd.DataFrame({'id': [1], 'y_true': [1], 'y_pred': [0]})
        b = pd.DataFrame({'id': [1], 'y_true': [0], 'y_pred': [0]})
        with self.assertRaises(ValueError):
            mcnemar_predictions(a, b)

    def test_croisement_preserve_annotations_et_identifie_contradiction(self):
        train = pd.DataFrame({'tweet': ['أهلا'], 'sarcasm': [0]})
        test = pd.DataFrame({'tweet': ['اهلا', 'سليم'], 'sarcasm': [1, 0]})
        erreurs = pd.DataFrame({'tweet': ['اهلا', 'سليم'],
            'categorie_principale': ['contexte', 'dialecte'],
            'justification': ['explication 1', 'explication 2']})
        r, conflits = croiser_erreurs_conflits(erreurs, train, test)
        self.assertEqual(r.doublon_contradictoire_4_1.tolist(), [True, False])
        self.assertEqual(len(conflits), 2)
        self.assertTrue(r[erreurs.columns].equals(erreurs))


if __name__ == '__main__':
    unittest.main()
