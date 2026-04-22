# 🔍 Retrievers - Systèmes de Récupération

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green.svg)](https://faiss.ai)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-purple.svg)](https://qdrant.tech)

> *Implémentations des différents systèmes de récupération utilisés dans le pipeline RAG pour extraire des informations pertinentes de la base de code*

---

## 📁 Contenu du dossier

Le dossier `rag/retrievers` contient les implémentations modulaires des systèmes de récupération :

### 📄 Fichiers principaux

- **`__init__.py`** - Module d'initialisation exportant `FAISSRetriever` et `GrepRetriever`
- **`faiss_retriever.py`** - Retriever basé sur FAISS pour recherche vectorielle
- **`grep_retriever.py`** - Retriever syntaxique utilisant grep pour fichiers DSL
- **`qdrant_retriever.py`** - Retriever hybride basé sur Qdrant (dense + sparse)

---

## 🎯 Types de retrievers

### 1. 🚀 FAISSRetriever

**Retriever basé sur FAISS** - Recherche de similarité vectorielle haute performance

#### ✨ Fonctionnalités

- 🔧 **Types d'index multiples** : `IndexFlatIP`, `IndexFlatL2`, `IndexIVFFlat`, `IndexHNSWFlat`
- ⚡ **Accélération GPU** optionnelle pour performances accrues
- 🔄 **Normalisation automatique** des embeddings pour similarité cosinus
- 🎯 **Déduplication intelligente** des résultats de recherche
- 💾 **Persistance disque** avec sauvegarde/chargement d'index

#### ⚙️ Configuration

```python
config = {
    'faiss': {
        'index_type': 'IndexFlatIP',      # IndexFlatIP, IndexIVFFlat, IndexHNSWFlat
        'use_gpu': False,                 # Accélération GPU
        'nlist': 100,                     # Nombre de clusters IVF
        'nprobe': 10,                     # Nombre de clusters à sonder
        'm': 16,                          # Connexions par nœud HNSW
        'ef_construction': 200,           # Qualité construction HNSW
        'ef_search': 64                   # Précision recherche HNSW
    },
    'files': {
        'chunks_filename': 'chunks',      # Nom du fichier chunks
        'index_filename': 'faiss',        # Nom du fichier index
        'metadata_filename': 'metadata'   # Nom du fichier métadonnées
    }
}
```

#### 💻 Utilisation

```python
from rag.retrievers import FAISSRetriever

# Initialisation
retriever = FAISSRetriever(config)
retriever.initialize(embedding_dimension=768)

# Indexation
retriever.add_chunks(chunks, embeddings)

# Recherche
results = retriever.search(query_embedding, top_k=10)

# Persistance
retriever.save_index()
retriever.load_index()
```

---

### 2. 🔍 GrepRetriever

**Retriever syntaxique** - Recherche textuelle précise dans les fichiers source

#### ✨ Fonctionnalités

- 🧠 **Recherche intelligente** de chemins de fichiers et références
- 🔤 **Support regex** pour patterns complexes
- 📄 **Spécialisé DSL** avec fichiers `.nvn` (Envision)
- 🗺️ **Mapping automatique** des chemins originaux
- 🔗 **Analyse de références** dans les scripts

#### 💻 Utilisation

```python
from rag.retrievers.grep_retriever import GrepRetriever

# Initialisation
retriever = GrepRetriever(search_dirs=['env_scripts/'])

# Recherche de pattern
results = retriever.search('pattern', case_sensitive=False)

# Recherche de chemins
results = retriever.search('/some/path/file.ext')
```

---

### 3. 🧠 QdrantRetriever

**Retriever hybride avancé** - Recherche sémantique + syntaxique avec Qdrant

#### ✨ Fonctionnalités

- 🎯 **Recherche dense** - Embeddings sémantiques haute qualité
- 📝 **Recherche sparse** - BM25 pour mots-clés
- 🔄 **Fusion RRF** - Reciprocal Rank Fusion pour résultats optimaux
- 🎛️ **Filtrage métadonnées** - Recherche par chemins de fichiers
- 📚 **Collections séparées** - Chunks complets vs résumés
- ☁️ **Support cloud/local** - Stockage flexible

#### ⚙️ Configuration

```python
config = {
    'collection_name': 'codebase_rag',     # Nom de la collection
    'rerank_multiplier': 2,                # Multiplicateur de classement
    'qdrant_path': './data/qdrant/',       # Chemin stockage local
    'local': True                          # Mode local vs cloud
}
```

#### 💻 Utilisation

```python
from rag.retrievers.qdrant_retriever import QdrantRetriever

# Initialisation
retriever = QdrantRetriever(config)
retriever.initialize(embedding_dimension=768)

# Indexation dense (rétrocompatible)
retriever.add_chunks(chunks, embeddings)

# Indexation hybride
retriever.add_chunks_hybrid(chunks, hybrid_embeddings)

# Recherche dense
results = retriever.search(query_embedding, top_k=10)

# Recherche hybride avancée
results = retriever.search_hybrid(
    query_text, embedder, top_k=10,
    source_substrings=['utils/']
)
```

---

## 🏗️ Architecture commune

Tous les retrievers héritent de `BaseRetriever` et implémentent l'interface unifiée :

### 🔧 Méthodes principales

- `initialize(embedding_dimension)` - Initialisation avec dimension d'embeddings
- `add_chunks(chunks, embeddings)` - Indexation de chunks avec embeddings
- `search(query_embedding, top_k)` - Recherche par embedding de requête
- `save_index() / load_index()` - Persistance sur disque
- `clear_index()` - Réinitialisation complète
- `get_chunk_count()` - Comptage des chunks indexés

---

## 🔄 Intégration dans le pipeline

Les retrievers s'intègrent parfaitement dans le pipeline RAG :

### 📋 Workflow typique

1. **📥 Indexation** - Transformation des chunks en représentations vectorielles
2. **🔍 Recherche** - Récupération des chunks les plus pertinents
3. **📤 Récupération** - Retour des résultats avec scores de similarité

### 🎯 Choix du retriever

| Cas d'usage | Recommandation | Avantages |
|-------------|----------------|-----------|
| **Performance pure** | FAISS | ⚡ Vitesse, 📊 Scalabilité |
| **Précision syntaxique** | Grep | 🎯 Exactitude, 🔍 Patterns complexes |
| **Recherche hybride** | Qdrant | 🧠 Sémantique + 🔤 Syntaxe, 🎛️ Filtrage avancé |

---

## 📦 Dépendances

### 🔧 Bibliothèques externes

- **`faiss-cpu/faiss-gpu`** - Pour FAISSRetriever (recherche vectorielle)
- **`qdrant-client`** - Pour QdrantRetriever (base vectorielle)
- **`numpy`** - Pour tous les retrievers (calculs vectoriels)

### 🔗 Modules internes

- `rag.core.base_retriever` - Classe de base commune
- `rag.core.base_chunker` - Gestion des chunks de code
- `rag.embedders.*` - Générateurs d'embeddings
- `config_manager` - Gestion de la configuration
