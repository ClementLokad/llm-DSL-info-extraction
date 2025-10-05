# 🚀 LLM DSL Information Extraction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green.svg)](https://faiss.ai)
[![AI](https://img.shields.io/badge/AI-GPT%20%7C%20Gemini%20%7C%20Mistral-orange.svg)](https://openai.com)

> *Un système sophistiqué d'analyse et d'interrogation de code DSL utilisant l'IA sémantique*

---

## 🎯 Vue d'ensemble du projet

Le **LLM DSL Information Extraction System** est une pipeline complète et modulaire conçue pour analyser, traiter et interroger intelligemment des codebases de langages spécifiques à un domaine (DSL), spécialement optimisée pour **Envision DSL** de Lokad.

### 🔥 Fonctionnalités clés

- 🔍 **Parsing intelligent** - Analyse sémantique des fichiers `.nvn` (Envision DSL)
- 🧩 **Chunking contextuel** - Segmentation intelligente préservant la cohérence
- 🎯 **Recherche vectorielle** - Index FAISS haute performance avec similarité cosinus
- 🤖 **Agents IA multiples** - Support GPT-4, Gemini, et Mistral avec rate limiting
- ⚙️ **Configuration externalisée** - Tous les paramètres dans `config.yaml`
- 🧪 **Tests exhaustifs** - Suite de tests par phases avec couverture complète
- 📊 **Interface CLI professionnelle** - Modes interactif, query, verbose, quiet
- 💾 **Sessions traçables** - Sauvegarde automatique des requêtes avec métriques

### 🎭 Cas d'usage

- **Exploration de codebase** - Comprendre rapidement une base de code DSL complexe
- **Documentation automatique** - Génération de documentation à partir du code
- **Assistance au développement** - Réponses contextuelles sur la logique métier
- **Analyse de dépendances** - Identification des relations entre composants

---

## 🏗️ Architecture système

```text
┌─────────────────────────────────────────────────────────────────┐
│                    🚀 DSL QUERY SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│  📋 CLI Interface (main.py, test.py, build_index.py)          │
├─────────────────────────────────────────────────────────────────┤
│  🤖 AI Agents Layer                                           │
│  ├── 🧠 GPT Agent      ├── 🔮 Gemini Agent   ├── ⚡ Mistral    │
├─────────────────────────────────────────────────────────────────┤
│  🔧 Processing Pipeline                                        │
│  ├── 📄 Parser  ├── 🧩 Chunker  ├── 🎯 Embedder  ├── 🔍 Retriever │
├─────────────────────────────────────────────────────────────────┤
│  💾 Data Layer                                                │
│  ├── 📁 Source Files    ├── 🗃️ FAISS Index    ├── 📊 Sessions  │
└─────────────────────────────────────────────────────────────────┘
```

### 📂 Structure du projet

```text
llm-DSL-info-extraction/
├── 🎮 main.py                    # Interface utilisateur principale
├── 🔨 build_index.py            # Construction d'index FAISS
├── 🧪 test.py                   # Suite de tests complète
├── ⚙️ config.yaml               # Configuration système
├── 🔧 config_manager.py         # Gestionnaire de configuration
├── 📄 requirements.txt          # Dépendances Python
├── 🔐 .env                      # Clés API (sensible)
├── 📖 README.md                 # Documentation principale
│
├── 🤖 agents/                   # Agents IA
│   ├── 📋 AGENTS.md             # Documentation agents
│   ├── 🔧 base.py               # Interface abstraite
│   ├── 🧠 gpt_agent.py          # Agent OpenAI GPT
│   ├── 🔮 gemini_agent.py       # Agent Google Gemini
│   ├── ⚡ mistral_agent.py      # Agent Mistral AI
│   └── 🧪 test_agents.py        # Tests des agents
│
├── 🔄 pipeline/                 # Pipeline de traitement
│   ├── 📋 PIPELINE.md           # Documentation pipeline
│   ├── 🏗️ core/                # Classes de base
│   │   ├── 📄 base_parser.py    # Interface parser
│   │   ├── 🧩 base_chunker.py   # Interface chunker
│   │   ├── 🎯 base_embedder.py  # Interface embedder
│   │   ├── 🔍 base_retriever.py # Interface retriever
│   │   └── 📊 session.py        # Gestion des sessions
│   ├── 📄 parsers/              # Analyseurs syntaxiques
│   │   └── 🔧 envision_parser.py # Parser Envision DSL
│   ├── 🧩 chunkers/             # Segmenteurs sémantiques
│   │   └── 🎯 semantic_chunker.py # Chunker contextuel
│   ├── 🎯 embedders/            # Générateurs d'embeddings
│   │   ├── 🤖 sentence_transformer_embedder.py
│   │   ├── 🧠 openai_embedder.py
│   │   └── 🔮 gemini_embedder.py
│   ├── 🔍 retrievers/           # Moteurs de recherche
│   │   └── ⚡ faiss_retriever.py # Recherche FAISS
│   └── 🧪 tests/               # Tests par composant
│
├── 💾 data/                     # Données et index
│   ├── 🗃️ faiss_index/         # Index FAISS sauvegardé
│   └── 📊 sessions/            # Sessions de requêtes
│
├── 📁 env_scripts/             # Fichiers source .nvn
└── 📄 env_txt/                 # Fichiers source .txt
```

---

## 🚀 Démarrage rapide

### 1. 📦 Installation

```bash
# Cloner le repository
git clone https://github.com/ClementLokad/llm-DSL-info-extraction.git
cd llm-DSL-info-extraction

# Créer et activer l'environnement virtuel
python -m venv env
.\env\Scripts\Activate.ps1  # Windows PowerShell
# source env/bin/activate    # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

### 2. ⚙️ Configuration

```bash
# Copier et configurer les variables d'environnement
cp .env.example .env

# Éditer .env avec vos clés API
GOOGLE_API_KEY=your-gemini-api-key-here
OPENAI_API_KEY=your-openai-api-key-here  
MISTRAL_API_KEY=your-mistral-api-key-here
```

### 3. 🔨 Construction de l'index de recherche

```bash
# Construction automatique (recommandée)
python build_index.py

# Options avancées
python build_index.py --force        # Force la reconstruction
python build_index.py --quiet        # Mode silencieux
python build_index.py --check        # Vérifier l'état de l'index
```

### 4. 🎮 Lancement du système

```bash
# Mode interactif (par défaut)
python main.py

# Requête directe
python main.py --query "What is MOV in the suppliers table?"

# Mode verbose avec détails
python main.py --verbose --query "explain business logic"

# Mode quiet pour scripts
python main.py --quiet --query "data flow"

# Vérifier le statut du système
python main.py --status
```

---

## 💻 Exemples d'utilisation

### 🎯 Mode interactif

```bash
python main.py --interactive
```

```text
🚀 INITIALIZING DSL QUERY SYSTEM
============================================================
🔧 Initializing pipeline components...
   ✅ Parser: ['.nvn']
   ✅ Chunker: max_tokens=512
   ✅ Embedder: model=all-MiniLM-L6-v2
   ✅ Retriever: metric=cosine
🤖 Initializing AI agent...
   ✅ Mistral-mistral-large-latest initialized
📚 Loading knowledge index...
✅ Index loaded: {vectors: 2847, chunks: 2847, dimensions: 384}
✅ System initialized and ready
============================================================

💬 DSL Query System - Interactive Mode
Type 'help' for commands, 'quit' to exit

👤 You: What is the MOV field in suppliers?
🤖 DSL: Based on the code analysis, MOV appears to be a numeric field 
representing "Minimum Order Value" - the smallest monetary amount 
required per supplier order. It's validated with `Suppliers.MOV > 0` 
to ensure positive values...
```

### 📊 Mode verbose avec analyse détaillée

```bash
python main.py --verbose --query "explain the business logic for customer tiers"
```

```text
🔍 PROCESSING QUERY
============================================================
📝 Query: explain the business logic for customer tiers
🤖 Agent: mistral
📊 Top-K chunks: 5
------------------------------------------------------------
🔤 Query embedding: (384,) dimensions (0.124s)
🔍 Retrieved 5 chunks in 0.001s

📋 RETRIEVED CHUNKS ANALYSIS:
================================================================================
[1] Score: 0.8432
    Content: when Sales.Revenue > 1000 then "Premium"
             when Sales.Revenue > 500 then "Gold"  
             when Sales.Revenue > 100 then "Silver"
             else "Bronze"
    Metadata: {'section': 'business_rules', 'type': 'conditional'}
[2] Score: 0.7891
    Content: Sales.CustomerTier = // Customer tier classification...
--------------------------------------------------------------------------------
📝 Prepared context from 5 chunks in 0.000s
🤖 Querying MistralAgent...
📄 Input prompt length: 1247 characters
💬 LLM responded in 2.341s

🎯 FINAL RESULTS:
================================================================================
📊 PROCESSING SUMMARY:
   • Query: explain the business logic for customer tiers
   • Retrieved chunks: 5
   • Context length: 1247 chars
   • Response length: 856 chars

⏱️ TIMING BREAKDOWN:
   • embedding: 0.124s
   • retrieval: 0.001s
   • context_preparation: 0.000s
   • llm_query: 2.341s
   • total: 2.471s

💬 LLM RESPONSE:
--------------------------------------------------------------------------------
The customer tier classification system uses a revenue-based hierarchy...
💾 Session saved to: data/sessions/session_20251005_143521.json
```

### 🧪 Tests et validation

```bash
# Tests complets du système
python test.py

# Tests par catégorie
python test.py --categories config,agents,pipeline

# Export des résultats
python test.py --export test_results.json

# Mode quiet pour CI/CD
python test.py --quiet
```

```text
🧪 DSL QUERY SYSTEM - COMPREHENSIVE TEST SUITE
================================================================
📊 Test Results Summary:
   • Config Tests: ✅ 2/2 passed (100.0%)
   • Agent Tests: ✅ 1/1 passed (100.0%)  
   • Pipeline Tests: ✅ 1/1 passed (100.0%)
   • Index Tests: ✅ 2/2 passed (100.0%)
   • Quality Tests: ✅ 2/2 passed (100.0%)
   • Integration Tests: ✅ 1/1 passed (100.0%)

🎯 OVERALL SUCCESS RATE: 9/9 tests passed (100.0%)
⏱️ Total execution time: 3.247s
💾 Results exported to: test_results.json
```

---

## 🔧 Configuration avancée

### 📋 Structure config.yaml

```yaml
# Configuration principale dans config.yaml
agent:
  default_model: "mistral"  # "gemini", "gpt", "mistral"

parser:
  type: "envision"
  supported_extensions: [".nvn"]

chunker:
  type: "semantic"
  max_chunk_tokens: 512
  strategies:
    group_by_section: true
    group_related_assignments: true

embedder:
  default_type: "sentence_transformer"
  sentence_transformer:
    model_name: "all-MiniLM-L6-v2"
    device: "cpu"  # "auto", "cpu", "cuda"

retriever:
  type: "faiss"
  faiss:
    index_type: "IndexFlatIP"  # Exact search
    top_k: 10
    search_threshold: 0.7
```

### 🎯 Optimisation des performances

```yaml
# Pour datasets volumineux (>100K chunks)
retriever:
  faiss:
    index_type: "IndexHNSWFlat"  # Approximate but fast
    hnsw_m: 16
    hnsw_ef_search: 64
    use_gpu: true  # Requires faiss-gpu

# Pour haute précision
retriever:
  faiss:
    index_type: "IndexFlatIP"  # Exact search
    search_threshold: 0.8  # Plus strict
```

---

## 🧠 Philosophie d'implémentation

### 🏗️ Architecture modulaire et évolutive

Le système suit une **architecture en couches** avec des **interfaces abstraites** permettant l'extension facile :

- **🔌 Extensibilité** - Nouveaux parsers, embedders, retrievers par simple héritage
- **⚙️ Configuration externalisée** - Tous les paramètres dans `config.yaml`
- **🧪 Test-driven** - Chaque composant testé individuellement et en intégration
- **📊 Observabilité** - Sessions traçables avec métriques détaillées
- **🔒 Sécurité** - Clés API isolées, validation des entrées

### 🎯 Patterns de conception utilisés

```text
🏗️ Strategy Pattern    → Agents IA interchangeables
🔧 Factory Pattern     → Création dynamique de composants  
📋 Template Method     → Pipeline de traitement standardisé
🎭 Adapter Pattern     → Intégration APIs diverses
📊 Observer Pattern    → Logging et métriques
🔌 Plugin Architecture → Extensions modulaires
```

---

## 🛠️ Guide de développement

### 🔧 Ajouter un nouvel agent IA

```python
# agents/claude_agent.py
from agents.base import LLMAgent, rate_limited

class ClaudeAgent(LLMAgent):
    def __init__(self, config: dict):
        self.api_key = config.get('claude_api_key')
        self.model = config.get('model', 'claude-3-sonnet')
  
    def initialize(self) -> None:
        # Initialiser le client Claude
        pass
  
    @rate_limited(max_retries=3)
    def generate_response(self, question: str, context: str = None) -> str:
        # Implémenter la logique de génération
        pass
```

### 🔧 Ajouter un nouveau type d'embedder

```python
# pipeline/embedders/custom_embedder.py
from pipeline.core.base_embedder import BaseEmbedder

class CustomEmbedder(BaseEmbedder):
    def initialize(self) -> None:
        # Charger le modèle custom
        pass
  
    def embed_text(self, text: str) -> np.ndarray:
        # Logique d'embedding personnalisée
        return embeddings
```

### 🔧 Étendre le parser pour un nouveau DSL

```python
# pipeline/parsers/sql_parser.py  
from pipeline.core.base_parser import BaseParser, CodeBlock

class SQLParser(BaseParser):
    @property
    def supported_extensions(self) -> List[str]:
        return [".sql"]
  
    def parse_content(self, content: str, file_path: str) -> List[CodeBlock]:
        # Parser SQL personnalisé
        pass
```

---

## 🧪 Tests et qualité

### 📊 Structure des tests

```text
🧪 Tests par phases :
├── 📄 Phase 1 - Parsing      : Extraction code blocks
├── 🧩 Phase 2 - Chunking     : Segmentation sémantique  
├── 🎯 Phase 3 - Embedding    : Génération embeddings
├── 🔍 Phase 4 - Retrieval    : Index et recherche
├── ⚙️ Configuration          : Validation paramètres
└── 🔗 Integration            : Tests end-to-end
```

### 🎯 Métriques de qualité

```bash
# Couverture des tests
python test.py --categories quality

# Performance benchmarks
python test.py --categories integration --verbose

# Validation configuration
python test.py --categories config
```

---

## 📈 Métriques et monitoring

### 📊 Sessions de query traçables

Chaque requête est automatiquement sauvegardée avec :

```json
{
  "session_id": "session_20251005_143521",
  "query": "What is MOV?",
  "timestamp": "2025-10-05T14:35:21",
  "retrieved_chunks": [...],
  "context_added": "...",
  "llm_response": "...",
  "timing": {
    "embedding": 0.124,
    "retrieval": 0.001, 
    "llm_query": 2.341,
    "total": 2.471
  },
  "steps": [...]
}
```

### ⚡ Métriques de performance

- **Embedding** : ~0.1s par requête (CPU)
- **Retrieval** : ~0.001s pour 1000 chunks (FAISS)
- **LLM Query** : 1-5s selon le modèle
- **Memory** : ~500MB pour index 1000 chunks

---

## 🚨 Troubleshooting

### ❌ Erreurs communes

```bash
# Erreur: sentence-transformers non installé
pip install sentence-transformers

# Erreur: Clé API manquante
# Vérifier .env et config.yaml

# Erreur: Index FAISS corrompu
python build_index.py --force

# Performance lente
# Utiliser GPU : pip install faiss-gpu
# Ou réduire la taille des chunks dans config.yaml
```

### 🔧 Debug et diagnostics

```bash
# Mode verbose pour debugging
python main.py --verbose --query "test"

# Vérifier état du système
python main.py --status

# Tests de santé complets
python test.py --categories integration
```

---

## 🤝 Contribution

### 📋 Guidelines

1. **🔧 Development** - Suivre les patterns existants
2. **🧪 Testing** - Tests pour chaque nouvelle feature
3. **📖 Documentation** - Mettre à jour les .md
4. **⚙️ Configuration** - Externaliser les paramètres

---

## 📝 Licence

Ce projet est sous licence PRIVATE LICENSE AGREEMENT. Voir [LICENSE](LICENSE) pour plus de détails.

---
