# 🚀 Tutoriel Setup Rapide

Ce guide te permet de mettre en place le système **LLM DSL Information Extraction** en quelques minutes, de zéro.

## 📋 Vue d'ensemble

Le processus de setup compte **5 étapes principales** :

1. Créer un environnement virtuel Python
2. Installer les dépendances du projet
3. Configurer les clés API (`.env`)
4. Construire l'index Qdrant
5. ✅ Setup terminé ! **Tu peux commencer à utiliser le système**

**Durée estimée :** 10-15 minutes (dont ~5 min pour la construction de l'index)

---

## Étape 1️⃣ : Créer un environnement virtuel

Un environnement virtuel isole les dépendances du projet de ton système Python global.

**Sur Windows :**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Sur macOS / Linux :**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Une fois activé, tu devrais voir `(.venv)` au début de ta ligne de terminal.

---

## Étape 2️⃣ : Installer les dépendances

Avec l'environnement virtuel activé, installe toutes les dépendances du projet :

```bash
pip install -r requirements.txt
```

**Ce qu'il s'installe :**
- **Embedders & vectorstore :** `fastembed`, `sentence-transformers`, `qdrant-client` (défaut), `faiss-cpu` (optionnel)
- **LLM APIs :** `mistralai`, `groq`, `anthropic`, `ollama` (pour local models)
- **Utilities :** `python-dotenv`, `pyyaml`, `tiktoken`, `langgraph`
- **Benchmarking :** `scikit-learn`, `rich`, `typer`

> **Note :** Si tu rencontres des problèmes de compilation, assure-toi d'avoir Python 3.9+ installé.

---

## Étape 3️⃣ : Configurer les clés API (.env)

### Copier le fichier exemple

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### Remplir la clé API

Ouvre le fichier `.env` créé et ajoute ta clé API **DeepSeek** (le modèle par défaut du projet) :

```yaml
# .env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

**Où récupérer ta clé DeepSeek ?**
1. Va sur [https://platform.deepseek.com/](https://platform.deepseek.com/)
2. Crée un compte ou connecte-toi
3. Va dans "API Keys" → "Create New API Key"
4. Copie ta clé et colle-la dans `.env`

### (Optionnel) Autres clés API

Si tu veux tester d'autres modèles LLM, tu peux aussi ajouter :

```yaml
MISTRAL_API_KEY=your_mistral_key_here
GROQ_API_KEY=your_groq_key_here
# etc.
```

Mais pour commencer, **seule la clé DeepSeek est nécessaire**.

---

## Étape 4️⃣ : (Optionnel) Configurer config.yaml

Le fichier `config.yaml` contient la configuration par défaut du système. **Les defaults fonctionnent très bien pour commencer**, mais tu peux les customiser si besoin (modèles LLM, type d'embedder, base vectorielle, etc.).

Pour le setup rapide, tu peux **ignorer cette étape** et garder les settings par défaut.

---

## Étape 5️⃣ : Construire l'index Qdrant

L'index Qdrant permet au système de récupérer rapidement les informations pertinentes des scripts DSL.

```bash
python build_summary_index.py
```

**Qu'est-ce qui se passe :**
1. Parse les fichiers `.nvn` du dossier `env_scripts/`
2. Génère des **summaries** (résumés sémantiques) pour chaque bloc de code
3. Embed les summaries
4. Construit l'index Qdrant dans `data/qdrant/`

**Durée :** ~5 minutes (dépend du nombre de scripts et de ta connexion)

> **Note :** Si tu veux utiliser **FAISS** à la place de Qdrant, configure `config.yaml` avec `embedder.type: faiss` et lance plutôt `python build_index.py` ou `python build_summary_index.py`.

Une fois terminé, tu verras un message ✅ confirmant que l'index a été créé.

---

## ✅ Setup terminé !

Bravo 🎉 ! Ton setup est maintenant **complet et opérationnel**.

Tu peux commencer à utiliser le système avec les fonctionnalités principales (voir section suivante).

---

## 🎯 Fonctionnalités principales

Le système a **3 modes principaux d'utilisation** : Query, Benchmark, et Interactive.

### Mode Query : Interroger le système

Pose une question unique et obtiens une réponse :

```bash
python main.py --query "Your question here"
```

**Exemples :**
```bash
# Simple query
python main.py --query "Comment fonctionne le processus de compilation ?"

# Avec output détaillé (trace le raisonnement pas-à-pas)
python main.py -v --query "Quels sont les modules critiques ?"

# Avec output minimal (juste la réponse finale)
python main.py -q --query "Liste-moi les variables globales"
```

**Output :** La réponse générée par le LLM, basée sur la recherche dans l'index.

---

### Mode Benchmark : Évaluer la performance

Évalue le système contre un ensemble de questions avec réponses de référence :

```bash
python main.py --benchmarkpath test.json
```

**Variantes :**
```bash
# Fichier de benchmark court (recommandé pour tester)
python main.py --benchmarkpath test.json

# Fichier plus complet (attention : peut prendre 10-30 min)
python main.py --benchmarkpath questions.json

# Avec résultats détaillés
python main.py -v --benchmarkpath test.json

# Sauvegarder les résultats dans un fichier JSON
python main.py -sd --benchmarkpath test.json

# Reprendre depuis la question N (utile si ça crash)
python main.py --benchmarkpath questions.json --benchmarkstart 5
```

**Output :** Score de performance (p.ex., similarité cosinus, jugement LLM, score hybrid, etc.).

---

### Mode Interactive : Conversation multi-requêtes

Lance une session interactive pour poser plusieurs questions sans relancer le script :

```bash
python main.py --interactive
```

**Exemple de session :**
```
👤 Envision Copilot (Ctrl+C to exit)
User: Que font les fonctions utilitaires ?
Assistant: [répond à ta question]

User: Peut-tu détailler le flux de données ?
Assistant: [continue à répondre dans la même session]

User: quit
[Fin de la session]
```

**Avantages :** Plus rapide si tu as plusieurs questions, l'index reste en mémoire.

---

## 🎛️ Options principales

Pour une liste **complète** de toutes les options, consulte l'aide intégrée :

```bash
python main.py --help
```

Ci-dessous, un tableau des options **les plus courantes** :

| Option | Court | Type | Description |
|--------|-------|------|-------------|
| `--interactive` | `-i` | flag | Lance le mode interactif (défaut) |
| `--query TEXT` | — | string | Lance une query unique puis quitte |
| `--verbose` | `-v` | flag | Affiche le flux de raisonnement détaillé |
| `--quiet` | `-q` | flag | Supprime les messages d'initialisation |
| `--agent NAME` | `-a` | choice | Change d'agent LLM : `mistral`, `groq`, `deepseek-v3`, `qwen`, `qwen-ssh` |
| `--indextype TYPE` | `-in` | choice | Type d'index : `full_chunk`, `summary` |
| `--querytransform MODE` | `-qt` | choice | Mode de transformation : `fusion`, `hyde`, `None` |
| `--benchmarkpath PATH` | `-bp` | path | Lance un benchmark avec les questions du fichier |
| `--benchmarktype TYPE` | `-bt` | choice | Type d'évaluation : `llm_as_a_judge`, `cosine_similarity`, `hybrid` |
| `--benchmarkagent AGENT` | `-ba` | choice | Agent LLM pour évaluation : `mistral`, `groq`, etc. |
| `--benchmarkstart N` | `-bs` | integer | Reprend le benchmark depuis la question N |
| `--save_data` | `-sd` | flag | Sauvegarde les résultats du benchmark en JSON |
| `--token_count` | `-tc` | flag | Affiche le nombre total de tokens utilisés |

**Exemples combinés :**
```bash
# Query avec agent Mistral et fusion
python main.py --agent mistral --querytransform fusion --query "ton query"

# Benchmark complet avec output détaillé
python main.py -v -bp questions.json -ba groq

# Interactive verbose avec index summary au lieu de full_chunk
python main.py -i -v --indextype summary
```

---

## 🔧 Troubleshooting

### ❌ "ModuleNotFoundError" après installation

**Solution :** Assure-toi que l'environnement virtuel est activé :
```bash
# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### ❌ "DEEPSEEK_API_KEY not found"

**Solution :** Vérifiez que le fichier `.env` existe et contient ta clé :
```bash
# Vérifier que le fichier existe
cat .env  # macOS/Linux
type .env # Windows

# Vérifier que la clé est présente
grep DEEPSEEK_API_KEY .env
```

### ❌ "Index not found" lors du lancement

**Solution :** Construis l'index d'abord :
```bash
python build_summary_index.py
```

### ❌ L'index met trop de temps à se construire

**Solution :** C'est normal pour la première construction (~5 min). Pour les reconstructions, tu peux ajouter `--rebuild` pour forcer une régénération.

### ❌ Erreur de connexion Qdrant

**Solution :** La configuration Qdrant par défaut utilise un mode in-memory. Si tu utilises un serveur Qdrant externe, configure `config.yaml` avec les bonnes coordonnées.

---

## 📚 Prochaines étapes

- **Consulte [README.md](README.md)** pour une documentation technique plus complète
- **Explore `config.yaml`** pour customiser les paramètres du système
- **Lance `python main.py --help`** pour voir toutes les options disponibles
- **Lis [methodologie.md](methodologie.md)** pour comprendre l'approche technique du système

---

**Bon développement ! 🎉**
