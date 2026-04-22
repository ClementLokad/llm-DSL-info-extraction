# Fiche Technique & Stack

Le projet **INF01** utilise des technologies de pointe en IA et gestion de données pour garantir performance et extensibilité.

## 🛠️ Stack Technologique

### 🧠 Intelligence Artificielle
- **LLMs** : Support multi-modèle via APIs (Claude 3.5 Sonnet, GPT-4o, Mistral Large 2).
- **Orchestration** : **LangGraph** pour la gestion des cycles et de la persistance de l'état.
- **RAG Framework** : Implémentation personnalisée située dans `rag/`.

### 📊 Base de Données & Indexation
- **Vector Store** : **FAISS** ou **Chroma** (au choix) pour le stockage des embeddings.
- **Embeddings** : Modèles `text-embedding-3-small` ou `bge-m3`.

### 🖥️ Backend & Scripts
- **Langage** : Python 3.10+
- **Analyse Statique** : Parser Envision personnalisé (basé sur regex et graphes de flux).
- **Visualisation** : TikZ (pour les flyers) et Mermaid (pour la doc).

## 🚀 Installation & Déploiement

Consultez le fichier [`README.md`](https://github.com/Kpihx/envision/blob/main/envision/README.md) à la racine du projet pour les instructions d'installation :

```bash
# Installation des dépendances
pip install -r requirements.txt

# Initialisation de l'index RAG
python build_index.py --path ./data/codebase

# Lancer l'agent
python main.py "Ma question sur Envision ?"
```

## 📉 Évaluation (Benchmarks)

Nous suivons les performances de l'agent via un système de benchmark intégré :
- **Questions** : Localisées dans `questions.json`.
- **Résultats** : Visualisables via `view_benchmark_results.py`.
- **Métrique** : Taux de succès sur l'extraction de faits exacts.
