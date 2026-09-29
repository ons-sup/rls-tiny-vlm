# Utilisation de l'IA

J'ai utilise Claude (Anthropic) tout au long du projet comme assistant de
programmation et pour comprendre le brief.

## Ce que Claude a fait
- Explication du brief et des concepts (VLM, attention, masque prefixe)
- Redaction du code initial de chaque fichier (tokenizer, dataset, encodeur
  CNN, attention multi-tete, decodeur, boucle d'entrainement, evaluation)
- Guidage pas a pas pour la mise en place de l'environnement (Git, GitHub,
  Google Colab, authentification par token)
- Debogage de plusieurs problemes rencontres en cours de route : fichiers
  tronques lors de copier-coller, bug de decalage (off-by-one) dans
  l'alignement des cibles d'entrainement decouvert grace a un exact-match de
  0% alors que la loss etait bonne
- Redaction du README et de ce fichier

## Ce que j'ai verifie moi-meme
- J'ai lu et compris chaque fichier ligne par ligne avec les explications
  fournies
- J'ai execute chaque etape moi-meme sur Google Colab, verifie les resultats
  a chaque fois (shapes des tenseurs, tests de correction, courbes de loss)
- J'ai repere et signale des resultats suspects (exact-match a 0%) qui ont
  mene a la decouverte et correction du bug d'alignement
- Je suis capable d'expliquer le role de chaque composant du modele
  (encodeur CNN, adapter, attention masquee, decodeur, boucle d'entrainement)
  et le sens de chaque metrique du tableau de resultats
