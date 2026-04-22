# Etat de l'art

## Le problème de départ

Les LLM sont très performants sur les langages généralistes présents massivement sur le web, mais cette force se dégrade rapidement sur des DSL propriétaires.

Dans un tel contexte, le modèle peut produire une réponse fluide tout en :

- inventant des chemins de fichiers ;
- confondant des conventions locales ;
- perdant la structure réelle du projet ;
- mélangeant logique métier et hypothèses plausibles.

## Pourquoi ne pas utiliser uniquement le prompting

Une stratégie fondée seulement sur un bon prompt reste trop fragile.

Elle peut fonctionner sur des questions simples, mais elle devient vite insuffisante lorsqu'il faut :

- retrouver un identifiant exact ;
- lier plusieurs scripts ;
- suivre un flux de données ;
- raisonner sur un dépôt long et structuré.

## Limites d'un RAG naïf

La recherche vectorielle seule apporte une bonne proximité sémantique, mais elle n'est pas suffisante pour des besoins d'ingénierie précis.

Elle répond mal :

- aux recherches exactes sur symboles ;
- aux chemins de fichiers ;
- aux occurrences syntaxiques précises ;
- aux questions structurelles sur les dépendances du dépôt.

## Vers une architecture hybride

Notre direction de travail a donc consisté à combiner plusieurs formes de recherche :

- **sémantique** pour la logique métier et les concepts ;
- **lexicale** pour les identifiants et motifs exacts ;
- **structurelle** pour les relations entre scripts, données, tables et fonctions.

Cette hybridation permet d'éviter qu'un seul paradigme de recherche devienne le point faible du système.

## Pourquoi une boucle agentique

Même avec plusieurs outils, une seule passe ne suffit pas toujours.

Les approches agentiques récentes sont intéressantes car elles permettent :

- de choisir l'outil adapté à la question ;
- d'itérer lorsqu'une première recherche est insuffisante ;
- de fonder la réponse finale sur des preuves collectées progressivement.

Dans notre projet, cette logique est particulièrement pertinente car la base de code Envision possède une structure riche, mais aussi coûteuse à explorer intégralement en une seule fois.
