# Etat de l'art

## Cadrage du probleme

Les grands modeles de langage (LLM) sont efficaces sur les langages de programmation courants mais
significativement moins fiables sur les DSL proprietaires. Dans ces contextes, les sorties du modele
peuvent etre fluides tout en etant syntaxiquement ou semantiquement invalides.

## Approches de reference

- **Approche par prompt uniquement :** cout faible, mais peu fiable pour les conventions DSL inconnues.
- **Fine-tuning :** potentiellement performant mais couteux, lent a mettre a jour et rigide operationnellement.
- **Approche RAG :** adaptative et plus facile a maintenir quand le code source et la documentation evoluent.

## Limites du RAG naif

La recherche vectorielle pure est efficace pour la proximite conceptuelle mais faible pour la recherche
exacte au niveau des symboles. Pour les taches d'ingenierie DSL, les contraintes lexicales exactes sont
souvent obligatoires.

## Direction agentique

Les paradigmes agentiques recents ameliorent la robustesse en combinant :

- selection d'outils ;
- planification iterative et retour en arriere ;
- synthese fondee sur les preuves avant emission de la reponse finale.

Cela motive l'architecture hybride et agentique adoptee dans Envision.
