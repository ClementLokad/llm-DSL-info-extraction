# Methodologie

## 1. Preparation des donnees

Le pipeline part des scripts source Envision et applique un pretraitement pilote par le parser
pour preserver la structure du DSL. Le decoupage (chunking) n'est pas purement base sur les tokens ;
il est concu pour conserver des unites de code semantiquement coherentes.

## 2. Strategie de recherche hybride

Le systeme combine deux modes de recherche complementaires :

- **Recherche semantique (RAG) :** pour les questions conceptuelles et l'exploration logique elargie.
- **Recherche lexicale (GREP) :** pour les chemins exacts, identifiants, variables ou contraintes syntaxiques.

Une etape de routage determine quel mode doit etre priorise pour chaque requete.

## 3. Orchestration agentique

Le workflow principal est implemente sous forme de pipeline agentique a base de graphe :

- planification et selection d'outils ;
- collecte iterative de preuves ;
- agregation du contexte ;
- generation et nettoyage de la reponse finale.

Le workflow supporte des re-essais controles et des boucles conditionnelles pour ameliorer la qualite des reponses.

## 4. Benchmarking et controle qualite

Le projet inclut des modes de benchmark et des workflows de notation, avec support pour :

- notation basee sur la similarite ;
- evaluation par modele-juge (LLM-as-a-Judge) ;
- suivi operationnel (latence, comportement de la recherche).

## 5. Configuration et reproductibilite

Le comportement d'execution est controle par une configuration centralisee (`config.yaml`), incluant :

- parametres de modele et limites de debit ;
- parametres de chunking et de recherche ;
- strategie d'indexation et modes de benchmark.
