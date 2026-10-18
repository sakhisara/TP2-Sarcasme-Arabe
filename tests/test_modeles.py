import importlib.util
import unittest

TORCH_DISPONIBLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_DISPONIBLE, 'PyTorch absent : tests des réseaux non exécutés.')
class TestModeles(unittest.TestCase):
    def test_attention_ignore_padding(self):
        import torch
        from src.models import PoolingAttentionC
        module = PoolingAttentionC(4).eval()
        etats = torch.randn(2, 3, 4)
        masque = torch.tensor([[True, True, False], [True, False, False]])
        contexte, alpha = module(etats, masque)
        self.assertEqual(tuple(contexte.shape), (2, 4))
        self.assertTrue(torch.all(alpha[~masque] == 0))
        self.assertTrue(torch.allclose(alpha.sum(1), torch.ones(2)))

    def test_architectures_sorties_et_padding(self):
        import numpy as np
        import torch
        from src.models import ModeleC
        from src.config import CONFIG_C
        poids = np.random.default_rng(42).normal(size=(10, 100)).astype('float32')
        ids = torch.tensor([[2, 3, 0], [4, 5, 6]])
        longueurs = torch.tensor([2, 3])
        for arch in ['C1', 'C2', 'C3', 'C4']:
            with self.subTest(arch=arch):
                config = {**CONFIG_C, 'arch': arch, 'max_len': 8}
                modele = ModeleC(poids, config).eval()
                with torch.no_grad():
                    logits = modele(ids, longueurs)
                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertTrue(torch.isfinite(logits).all())


if __name__ == '__main__':
    unittest.main()
