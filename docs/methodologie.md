# Methodologie

## Vue d'ensemble

Le système final repose sur une architecture **hybride et agentique** conçue pour explorer une base de code Envision sans exposer directement tout le dépôt au modèle.

L'idée centrale est simple :

1. préparer le code sous une forme exploitable par des outils de recherche spécialisés ;
2. laisser un planificateur choisir le bon outil selon la nature de la question ;
3. agréger les preuves obtenues avant génération de la réponse finale.

## 1. Préparation des scripts Envision

Les scripts `.nvn` ne sont pas découpés arbitrairement.

Un parseur spécifique à Envision identifie des blocs de code cohérents :

- imports ;
- lectures de données (`read`) ;
- écritures (`write`) ;
- définitions de tables ;
- exports et fonctions ;
- zones de restitution (`show`) ;
- blocs de commentaires structurants.

Cette étape est importante, car elle permet ensuite :

- au RAG de travailler sur des chunks plus sémantiques ;
- au grep tool de cibler certaines structures syntaxiques ;
- au graph tool de reconstruire des relations plus précises entre scripts, tables, fonctions et fichiers.

## 2. Trois outils complémentaires

### RAG tool

Le RAG tool sert aux questions conceptuelles ou métier.  
Il récupère des extraits sémantiquement proches de la requête, même si la formulation exacte n'apparaît pas telle quelle dans le code.

### Grep tool

Le grep tool sert aux recherches exactes :

- noms de fonctions ;
- identifiants ;
- chemins de fichiers ;
- motifs syntaxiques précis.

Il ne travaille pas sur du texte brut uniquement, mais sur les blocs Envision déjà parsés.  
Il peut aussi restreindre la recherche à un dossier ou à un type de bloc (`READ`, `SHOW`, etc.).

### Graph tool

Le graph tool construit une représentation structurelle du dépôt :

- scripts ;
- données ;
- dossiers ;
- tables ;
- fonctions ;
- relations `imports`, `reads`, `writes`, `defines`, `contains`, etc.

Il est particulièrement utile pour répondre à des questions du type :

- quels scripts lisent un fichier donné ;
- quel module est importé par tel script ;
- dans quel script une table ou une fonction est définie.

## 3. Boucle agentique

Le système n'est pas une pipeline linéaire.

Un planificateur :

- lit la question ;
- choisit un outil ;
- observe le résultat ;
- décide soit de répondre, soit de poursuivre l'exploration avec une nouvelle requête plus ciblée.

Cette boucle donne au système un comportement plus proche d'une navigation réelle dans le code que d'une simple recherche statique.

## 4. Contrôle de qualité

Le projet inclut aussi une couche de benchmarking et de validation :

- benchmarks sur jeux de questions/réponses ;
- métriques automatiques ;
- évaluation par modèle juge ;
- contrôle de certaines hallucinations, notamment sur les chemins de scripts cités dans les réponses finales.

## 5. Reproductibilité

Le comportement du système est piloté par `config.yaml`.

Cette configuration centralise notamment :

- les modèles utilisés ;
- les paramètres de recherche ;
- les index ;
- les limites de la boucle agentique ;
- certains mécanismes de validation des réponses.
