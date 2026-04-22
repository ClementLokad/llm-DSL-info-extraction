# Progres depuis le rapport intermediaire

## Ce qui etait deja pose

Au stade intermédiaire, le projet avait déjà identifié :

- la difficulté spécifique des DSL propriétaires pour les LLM ;
- l'intérêt d'une recherche hybride ;
- la nécessité d'une orchestration plus riche qu'une simple chaîne `question -> retrieval -> réponse`.

## Ce qui a réellement évolué

Depuis ce stade, le projet s'est nettement structuré autour d'une architecture agentique plus aboutie.

Les principales évolutions sont les suivantes :

- consolidation du planificateur et de la boucle d'appels d'outils ;
- raffinement du RAG et de la recherche hybride ;
- ajout d'un `grep_tool` plus précis, appuyé sur les blocs syntaxiques Envision ;
- ajout d'un `graph_tool` capable de reconstruire explicitement les dépendances du dépôt ;
- meilleure gestion des chemins logiques via le mapping et les outils de navigation ;
- amélioration du benchmarking et du contrôle qualité.

## Changement de maturité

Le projet est passé d'un prototype centré sur le retrieval à un système plus cohérent au niveau ingénierie :

- séparation plus nette des outils ;
- configuration centralisée ;
- pipeline agentique modulaire ;
- meilleure traçabilité des preuves collectées ;
- premiers garde-fous contre certaines hallucinations dans les réponses finales.

## Enjeu actuel

L'enjeu n'est plus seulement de retrouver un extrait pertinent, mais de permettre au système de choisir la bonne stratégie d'exploration selon la question :

- sémantique ;
- lexicale ;
- structurelle.

C'est ce basculement qui marque la principale différence entre l'état intermédiaire et l'état actuel du projet.
