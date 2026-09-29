# rls-tiny-vlm

Un petit Vision-Language Model qui regarde une image (64x64, une ou deux formes
colorees) et ecrit le mot qui la decrit, une lettre a la fois.

## Architecture

- Encodeur CNN : 3 couches (32-64-128 canaux, stride 2) -> 8x8 = 64 tokens visuels
- d_model=128, 4 tetes d'attention, 2 couches de decodeur, FFN=256
- Masque prefixe : tokens visuels bidirectionnels entre eux, lettres causales,
  lettres voient toujours tout le visuel
- ~1.2M parametres
- Attention multi-tete ecrite a la main (voir src/model/attention.py),
  validee par un test d'equivalence avec F.scaled_dot_product_attention
  (difference max : 6.7e-08, tests/test_attention.py)

## Comment reproduire les resultats

git clone https://github.com/ons-sup/rls-tiny-vlm.git
cd rls-tiny-vlm
pip install Pillow tqdm numpy
python generate_data.py
python -m tests.test_shapes
python -m tests.test_attention
python -m src.train --mode e0
python -m src.train --mode full --epochs 10
python -m src.run_eval --checkpoint checkpoint_epoch9.pt --split test
python -m src.train_blind --epochs 10
python -m src.run_eval --checkpoint checkpoint_blind_epoch9.pt --split test
python -m benchmarks.throughput

## Resultats

### E0 - sanity check (overfitting d'un batch)

| step | 0 | 50 | 100 | 150 | 200 | 250 | final |
|---|---|---|---|---|---|---|---|
| loss | 3.4512 | 1.5293 | 0.5863 | 0.0839 | 0.0241 | 0.0131 | 0.0088 |

### Entrainement complet (10 epochs, Colab T4)

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| train loss | 0.5460 | 0.0800 | 0.0521 | 0.0423 | 0.0347 | 0.0329 | 0.0262 | 0.0236 | 0.0220 | 0.0207 |
| val loss | 0.1208 | 0.0659 | 0.3131 | 0.0440 | 0.0374 | 0.0326 | 0.0300 | 0.0423 | 0.0880 | 0.0313 |

### Tableau de resultats principal

| Experiment | Configuration | Metric | Result | Interpretation |
|---|---|---|---|---|
| Main model | 10 epochs, batch 32 | Exact match, test | 68.2% | Le modele apprend correctement a decrire les scenes |
| E1 blind | 10 epochs, images = 0 | Exact match, test | 0.0% | Sans image, le modele ne peut pas deviner le mot exact |
| E1 blind | idem | Val loss finale | 1.3475 (vs 0.0313 normal) | ~43x pire : preuve que le modele utilise vraiment l'image |
| S1 throughput | batch 64, Colab T4 GPU | images/s | 5500.6 | Debit eleve pour un modele de cette taille |

### Accuracy par attribut - modele normal vs baseline aveugle (E1)

| Attribut | Modele normal | Baseline aveugle |
|---|---|---|
| size | 88.25% | 49.75% |
| color | 80.55% | 30.60% |
| shape | 81.40% | 24.10% |
| relation | 41.86% | 2.55% |
| size2 | 75.78% | 4.02% |
| color2 | 59.90% | 2.25% |
| shape2 | 61.86% | 0.98% |

Observation principale : les relations spatiales sont l'attribut le plus difficile
a apprendre pour ce modele (41.9%, contre 80%+ pour couleur/forme/taille du premier
objet). Les attributs du 2e objet sont aussi systematiquement moins bons que ceux
du 1er objet, coherent avec une accumulation d'erreurs au fil de la generation
lettre par lettre.

Sur la baseline aveugle, size reste a ~50% meme sans image : logique, il n'y a
que 2 valeurs possibles (small/large), donc deviner au hasard suffit a atteindre
ce score par pur hasard.

### S1 - benchmark de debit (Colab T4 GPU)

| batch_size | images/sec |
|---|---|
| 1 | 140.2 |
| 16 | 2109.0 |
| 64 | 5500.6 |
| 256 | 6804.5 |

Le debit plafonne entre batch 64 et 256, signe qu'on approche de la limite de
calcul du GPU T4 pour ce modele a cette taille de batch.

### Test complementaire : 15 epochs au lieu de 10

Un second entrainement a 15 epochs a ete teste pour comparer. Resultat : la val
loss finale est pire (0.1968 a l'epoch 14, contre 0.0313 a l'epoch 9), avec un
minimum atteint plus tot, vers l'epoch 10 (val loss 0.0283). Ceci confirme un
sur-apprentissage progressif au-dela de 10 epochs. Le modele retenu comme
resultat principal reste donc celui entraine sur 10 epochs.
