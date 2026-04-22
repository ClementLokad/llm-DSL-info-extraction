# 🔮 Embedders - Générateurs d'Embeddings

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com)
[![Google](https://img.shields.io/badge/Google-Gemini-blue.svg)](https://ai.google.dev)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Local-orange.svg)](https://sbert.net)

> *Générateurs d'embeddings pour transformer les chunks de code en représentations vectorielles sémantiques*

---

## 📁 Contenu du dossier

Le dossier `rag/embedders` contient les différents générateurs d'embeddings :

### 📄 Fichiers principaux

- **`__init__.py`** - Module d'initialisation avec imports conditionnels
- **`sentence_transformer_embedder.py`** - Embedder local haute performance
- **`openai_embedder.py`** - Embedder OpenAI avec gestion de quota
- **`gemini_embedder.py`** - Embedder Google Gemini alternatif
- **`qdrant_embedder.py`** - Embedder hybride dense + sparse

---

## 🎯 Types d'embedders

### 1. 🏠 SentenceTransformerEmbedder

**Embedder local haute performance** - Modèles sentence-transformers sans API

#### ✨ Fonctionnalités

- 🚀 **Traitement local** - Aucun appel API, pas de quota
- 📊 **Volume élevé** - Idéal pour traiter de gros volumes de code
- 🔧 **Modèles flexibles** - Support de nombreux modèles pré-entraînés
- 💾 **Cache local** - Téléchargement et sauvegarde automatique des modèles
- ⚡ **Accélération GPU** - Support optionnel du GPU

#### ⚙️ Configuration

```python
config = {
    "general": {
        "sentence_transformer": {
            "model_name": "all-MiniLM-L6-v2",    # Modèle à utiliser
            "model_path": "data/sentence_transformer",  # Chemin local
            "device": null,                      # null = auto-détection
            "trust_remote_code": false,          # Sécurité du code distant
            "show_progress_bar": true,           # Barre de progression
            "convert_to_numpy": true             # Format de sortie
        }
    }
}
```

#### 💻 Utilisation

```python
from rag.embedders import SentenceTransformerEmbedder

# Initialisation
embedder = SentenceTransformerEmbedder(config)
embedder.initialize()

# Embedding de chunks
embeddings = embedder.embed_chunks(chunks)

# Embedding de texte simple
embedding = embedder.embed_text("def hello_world():")
```

---

### 2. 🤖 OpenAIEmbedder

**Embedder OpenAI haute qualité** - API embeddings avec gestion de quota

#### ✨ Fonctionnalités

- 🎯 **Qualité supérieure** - Meilleure compréhension sémantique
- 📏 **Dimensions variables** - 1536 ou 3072 dimensions selon le modèle
- 🔄 **Gestion de quota** - Rate limiting automatique
- 🔁 **Retry automatique** - Gestion des erreurs temporaires
- 📊 **Suivi d'usage** - Compteurs de requêtes et tokens

#### ⚙️ Configuration

```python
config = {
    "model_name": "text-embedding-3-small",     # Modèle OpenAI
    "api_key": null,                           # null = variable d'environnement
    "requests_per_minute": 3000,               # Limite requêtes/minute
    "tokens_per_minute": 1000000,              # Limite tokens/minute
    "max_retries": 3,                          # Nombre de tentatives
    "retry_delay": 1.0                         # Délai entre tentatives
}
```

#### 💻 Utilisation

```python
from rag.embedders.openai_embedder import OpenAIEmbedder

# Initialisation (avec clé API)
embedder = OpenAIEmbedder(config)
embedder.initialize()

# Embedding avec gestion automatique du quota
embeddings = embedder.embed_chunks(chunks)
```

---

### 3. 🌟 GeminiEmbedder

**Embedder Google Gemini** - Alternative haute qualité à OpenAI

#### ✨ Fonctionnalités

- 🎯 **Qualité sémantique** - Bonne compréhension du code
- 🔄 **Gestion de quota** - Rate limiting adapté à Gemini
- 🔁 **Retry intelligent** - Gestion des erreurs API
- 📏 **Dimension fixe** - 768 dimensions pour les modèles actuels
- ☁️ **Cloud-native** - Intégration Google Cloud

#### ⚙️ Configuration

```python
config = {
    "model_name": "models/text-embedding-004",  # Modèle Gemini
    "api_key": null,                           # null = variable d'environnement
    "requests_per_minute": 1500,               # Limite Gemini
    "max_retries": 3,                          # Tentatives de retry
    "retry_delay": 1.0                         # Délai entre tentatives
}
```

#### 💻 Utilisation

```python
from rag.embedders.gemini_embedder import GeminiEmbedder

# Initialisation
embedder = GeminiEmbedder(config)
embedder.initialize()

# Embedding de haute qualité
embeddings = embedder.embed_chunks(chunks)
```

---

### 4. 🔄 QdrantEmbedder

**Embedder hybride avancé** - Dense + Sparse pour recherche optimale

#### ✨ Fonctionnalités

- 🎯 **Embeddings denses** - Sémantique via sentence-transformers
- 📝 **Embeddings sparse** - BM25 pour recherche par mots-clés
- 🔄 **Fusion optimale** - Combinaison dense + sparse dans Qdrant
- ⚡ **Performance ONNX** - Accélération via runtime ONNX
- 🎛️ **Configuration flexible** - Modèles interchangeables

#### ⚙️ Configuration

```python
config = {
    "dense_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "sparse_model_name": "Qdrant/bm25",
    "disable_stemmer": true,                    # Désactiver le stemming
    "embedding_dimension": 384                 # Dimension dense
}
```

#### 💻 Utilisation

```python
from rag.embedders.qdrant_embedder import QdrantEmbedder

# Initialisation
embedder = QdrantEmbedder(config)
embedder.initialize()

# Embedding standard (dense uniquement)
embeddings = embedder.embed_chunks(chunks)

# Embedding hybride pour Qdrant
hybrid = embedder.embed_text_hybrid("chercher fonction", ["def", "function"])
# Retourne: {"dense": [...], "sparse": SparseVector(...)}
```

---

## 🏗️ Architecture commune

Tous les embedders héritent de `BaseEmbedder` et implémentent l'interface unifiée :

### 🔧 Méthodes principales

- `initialize()` - Initialisation du modèle/API
- `embed_chunks(chunks)` - Embedding d'une liste de chunks
- `embed_text(text)` - Embedding d'un texte simple
- `embed_batch(texts)` - Embedding par lot
- `embedding_dimension` - Dimension des vecteurs produits

### 📦 Interface standardisée

```python
class BaseEmbedder:
    def initialize(self) -> None: ...
    def embed_chunks(self, chunks: List[CodeChunk]) -> np.ndarray: ...
    def embed_text(self, text: str) -> np.ndarray: ...
    def prepare_chunk_for_embedding(self, chunk: CodeChunk) -> str: ...
    def prepare_text_for_embedding(self, text: str) -> str: ...
```

---

## 🔄 Intégration dans le pipeline

Les embedders s'intègrent parfaitement dans le pipeline RAG :

### 📋 Workflow typique

1. **📄 Parsing** - Extraction des blocs de code sémantiques
2. **✂️ Chunking** - Découpage en chunks de taille optimale
3. **🔮 Embedding** - Transformation en vecteurs denses/sparse
4. **💾 Indexation** - Stockage dans la base vectorielle
5. **🔍 Recherche** - Récupération par similarité vectorielle

### 🎯 Choix de l'embedder

| Cas d'usage | Recommandation | Avantages | Inconvénients |
|-------------|----------------|-----------|---------------|
| **Production haute volumétrie** | SentenceTransformer | ⚡ Rapide, 📦 Local, 💰 Gratuit | 📊 Qualité moyenne |
| **Qualité maximale** | OpenAI | 🎯 Excellente sémantique, 🔧 Fiable | 💰 Coût API, 📊 Quota |
| **Alternative économique** | Gemini | 🎯 Bonne qualité, ☁️ Cloud | 📊 Quota Google |
| **Recherche hybride** | QdrantEmbedder | 🔄 Dense + Sparse, ⚡ Performant | 🔧 Configuration complexe |

---

## 📦 Dépendances

### 🔧 Bibliothèques externes

- **`sentence-transformers`** - Pour SentenceTransformerEmbedder (local)
- **`openai`** - Pour OpenAIEmbedder (API)
- **`google-generativeai`** - Pour GeminiEmbedder (API)
- **`fastembed`** - Pour QdrantEmbedder (hybride)
- **`numpy`** - Pour tous les embedders (calculs vectoriels)

### 🔗 Modules internes

- `rag.core.base_embedder` - Classe de base abstraite
- `rag.core.base_chunker` - Gestion des chunks de code
- `config_manager` - Configuration centralisée et clés API

---

## ⚠️ Gestion des quotas et limites

### 📊 Limites par fournisseur

| Fournisseur | Requêtes/minute | Tokens/minute | Coût approximatif |
|-------------|-----------------|---------------|-------------------|
| **OpenAI** | 3,000 | 1,000,000 | $0.02/1M tokens |
| **Gemini** | 1,500 | Variable | $0.01/1M tokens |
| **Local** | ∞ | ∞ | $0 (électricité) |

### 🛡️ Bonnes pratiques

- **🔄 Rate limiting** - Respect automatique des limites API
- **💾 Cache** - Stockage local des modèles téléchargés
- **🔁 Retry** - Gestion automatique des erreurs temporaires
- **📊 Monitoring** - Suivi de l'usage et des coûts
- **⚡ Batching** - Traitement par lots pour optimiser les appels API

---

## 🔧 Extension pour nouveaux modèles

Pour ajouter un nouvel embedder :

```python
from rag.core.base_embedder import BaseEmbedder
import numpy as np

class NewEmbedder(BaseEmbedder):
    def initialize(self) -> None:
        # Initialisation du modèle
        self.model = load_your_model()
        self._is_initialized = True
    
    def _embed_batch_impl(self, texts: List[str]) -> np.ndarray:
        # Implémentation spécifique
        return np.array([self.model.encode(text) for text in texts])
```

Puis ajouter au `__init__.py` :
```python
try:
    from rag.embedders.new_embedder import NewEmbedder
except ImportError:
    NewEmbedder = None
```