# Workflow Agentique

Le processus de résolution d'une question technique suit une boucle **ReAct** (Reasoning + Acting) optimisée pour le code.

## 🔄 Cycle de Résolution

L'agent ne répond pas immédiatement. Il suit une séquence d'étapes itératives :

1.  **Planification** : Analyse la question et définit les outils nécessaires.
2.  **Exécution** : Appelle un ou plusieurs outils (ex: chercher une définition dans le RAG).
3.  **Observation** : Lit le résultat de l'outil (un extrait de code, une liste de dépendances).
4.  **Raffinement** : Met à jour son plan ou demande une précision si nécessaire.

## 🛠️ Les Outils de l'Agent

Les outils sont implémentés dans [`pipeline/agent_workflow/`](https://github.com/Kpihx/envision/tree/main/envision/pipeline/agent_workflow/).

### `rag_tool`
- **Source** : `rag_tool.py`
- **Rôle** : Recherche sémantique dans la documentation Envision et les commentaires de code.
- **Usage** : "Qu'est-ce qu'une table `O` dans Envision ?"

### `grep_tool`
- **Source** : `grep_tool.py`
- **Rôle** : Recherche textuelle exacte (regex).
- **Usage** : "Où est définie la variable `totalAmount` ?"

### `graph_tool`
- **Source** : `graph_tool.py`
- **Rôle** : Navigation dans la structure de dépendance.
- **Usage** : "Quels fichiers appellent ce script ?"

## ⚖️ Validation et Qualité

Un composant critique est le **Grader** ([`pipeline/answer_validation.py`](https://github.com/Kpihx/envision/tree/main/envision/pipeline/answer_validation.py)).

Avant d'afficher la réponse à l'utilisateur, un second LLM (le "Judge") vérifie :
- **Hallucination** : La réponse est-elle étayée par les preuves extraites ?
- **Pertinence** : Répond-elle exactement à la question ?
- **Qualité** : Le code cité est-il correct ?

Si le Grader échoue, l'agent reçoit un feedback et doit recommencer sa recherche avec une nouvelle stratégie.
