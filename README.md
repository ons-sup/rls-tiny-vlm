# rls-tiny-vlm — point de départ

Architecture (à ajuster si besoin, mais justifier dans le rapport) :
- CNN 3 couches (32→64→128 canaux, stride 2) → 8×8 = **64 tokens visuels**
- `d_model=128`, 4 têtes, 2 couches de décodeur, FFN=256
- Masque préfixe : tokens visuels bidirectionnels entre eux, lettres causales, lettres voient toujours tout le visuel
- ~1.2M paramètres

## Étape 0 — à faire avant tout le reste

`src/data.py` contient un `load_split()` **non implémenté**. Ouvrez le vrai
`generate_data.py` fourni par RLS, regardez ce qu'il écrit sur disque
(dossier d'images ? fichier `.npz` ? tensor `.pt` ?), puis remplissez
`load_split()` en conséquence. Rien d'autre ne dépend de ce détail.

## Ordre d'exécution recommandé

```bash
# 1. Vérifier le tokenizer seul
python -m src.tokenizer

# 2. Vérifier l'encodeur CNN seul (juste les formes)
python -m src.model.encoder

# 3. Vérifier l'attention seule
python -m src.model.attention

# 4. Vérifier les formes de bout en bout
python -m tests.test_shapes

# 5. Test de correction obligatoire : attention vs SDPA
python -m tests.test_attention

# 6. E0 — overfitter un batch (bug check avant tout entraînement réel)
python -m src.train --mode e0

# 7. Entraînement complet
python -m src.train --mode full --epochs 10

# 8. Benchmark de vitesse (S1, obligatoire)
python -m benchmarks.throughput

# 9. Vérifier le parser d'évaluation seul
python -m src.evaluate
```

## Ce qui manque encore (à vous de le construire)

- Le chargement réel des données (`load_split`)
- `generate.py` en script séparé si vous préférez (la logique existe déjà dans `TinyVLM.generate`)
- `configs/baseline.yaml` et `configs/blind.yaml` (E1 : même modèle, images remplacées par des zéros)
- `experiments/E1_blind/notes.md` et `benchmarks/S1_throughput/hardware.txt`
- `AI_USAGE.md`

## Résultats (à remplir)

| Experiment | Configuration | Metric | Result | Interpretation |
|---|---|---|---|---|
| Main model | configs/baseline.yaml | Exact match, test | … | … |
| E1 blind | configs/blind.yaml | Attribute acc., test | … | … |
| S1 throughput | batch 64 | images/s | … | … |
