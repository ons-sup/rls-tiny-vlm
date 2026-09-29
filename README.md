### Test complementaire : 15 epochs au lieu de 10

Un second entrainement a 15 epochs a ete teste pour comparer. Resultat : la val
loss finale est pire (0.1968 a l'epoch 14, contre 0.0313 a l'epoch 9), avec un
minimum atteint plus tot, vers l'epoch 10 (val loss 0.0283). Ceci confirme un
sur-apprentissage progressif au-dela de 10 epochs. Le modele retenu comme
resultat principal reste donc celui entraine sur 10 epochs.
