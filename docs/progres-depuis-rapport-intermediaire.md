# Progres depuis le rapport intermediaire

## Etat de reference au stade intermediaire

Au stade intermediaire, le projet avait deja valide :

- le defi specifique au DSL et la necessite du RAG ;
- les principes de recherche hybride ;
- l'orchestration agentique initiale et le cadrage du benchmark.

## Progres observes sur la branche main

En se basant sur la branche main actuelle, l'implementation consolide et etend desormais l'architecture :

- workflow d'appel d'outils renforce avec logique de planification structuree ;
- route explicite entre recherche conceptuelle et recherche exacte ;
- raffinements de la recherche incluant reranking et scoring tenant compte de la source ;
- etape de nettoyage des reponses avec contraintes de formatage strictes ;
- variantes du pipeline de benchmark pour plusieurs modes d'evaluation ;
- configuration operationnelle enrichie et support de changement de modele.

## Ameliorations de maturite ingenierie

Le projet presente desormais des traits de maturite plus proches de la production :

- modularisation plus claire (`pipeline`, `rag`, `agents`) ;
- controles d'execution configurables (comportement latence/debit) ;
- meilleure tracabilite de l'execution et de l'agregation de preuves ;
- controles de coherence ameliores dans les boucles de generation iterative.
