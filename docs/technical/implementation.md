# Implémentation technique

## Détails de l'intégration

L'intégration d'Envision Copilot suit une approche modulaire où chaque outil est une "skill" isolée accessible par l'agent.

### Gestion du Contexte
Pour éviter le dépassement de la fenêtre de contexte des LLMs, nous utilisons un système de **récurrence de résumé** et de **distillation de preuves**. Seuls les extraits de code les plus pertinents sont conservés pour la génération finale.

### Analyse du DSL Envision
Le DSL Envision possédant des spécificités sémantiques (tiles, vectors, etc.), nous avons implémenté des parsers personnalisés pour enrichir les métadonnées lors de l'indexation.
