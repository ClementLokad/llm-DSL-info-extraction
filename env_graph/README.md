# Envision Graph (`env_graph`)

Graph extraction and navigation utilities for the `envision/` repository.

This package parses Envision DSL scripts (`.nvn`) and builds a hierarchical
dependency graph with two domains: scripts and data files.

---

## 🚀 Quick Start

```bash
# 1. Build the dependency graph
python -m env_graph.network --build

# 2. Explore the structure
python -m env_graph.network --tree /

# 3. Show statistics
python -m env_graph.network --stats
```

---

## 📋 CLI Reference

```
python -m env_graph.network [OPTIONS]

Construction:
  --build               Construire le graphe depuis les scripts

Navigation:
  --tree [PATH]         Afficher l'arborescence (défaut: racine)
  --domain DOMAIN       scripts | data | both (défaut: scripts)

Statistiques:
  --stats               Afficher les statistiques du réseau
  --type TYPE           Filtrer par type de nœud
  --edge-type TYPE      Filtrer par type d'arête

Contenu:
  --read NODE_ID        Lire le contenu d'un nœud
  -l, --lines RANGE     Plage de lignes (ex: 1-50)
  --grep PATTERN        Rechercher un pattern regex

Exploration:
  --node NODE_ID        Détails d'un nœud
  --search QUERY        Rechercher par nom/chemin
  --neighbors NODE_ID   Explorer les voisins
  -d, --direction DIR   incoming | outgoing | siblings | all
  -r, --relation TYPE   Filtrer par type de relation

```

---

## ⚙️ Configuration (`config.yaml`)

Project-level overrides live in [config.yaml](/home/kpihx/Work/X/envision/envision/config.yaml)
under the `env_graph:` section. Package defaults remain in
[env_graph/config.yaml](/home/kpihx/Work/X/envision/envision/env_graph/config.yaml).

### Mode Lite vs Full

| Mode | Description | Économie Tokens |
|------|-------------|-----------------|
| `lite` | Minimal, stats préservés | **-80%** sur `get_tree` |
| `full` | Toutes les métadonnées | Pour debug |

---

## 🏗️ Architecture

```text
env_graph/
├── api.py          # Public API (EnvisionGraphAPI)
├── builder.py      # Graph construction
├── extractor.py    # Symbol extraction
├── typedefs.py     # Node / edge types
├── network.py      # CLI entry point
├── utils.py        # Config + mapping helpers
└── config.yaml     # Package defaults
```

### Deux Domaines de Dossiers

```
SCRIPTS                          DATA
=======                          ====
📁 /                             📁 /
├── 📁 /1. utilities             ├── 📁 /Input
│   └── 📜 67982                 │   └── 📄 Catalog.csv
└── 📁 /3. Inspectors            └── 📁 /Clean
    └── 📜 68006                     └── 📄 Items.ion
```

---

## 📊 Modèle de Données

### Types de Nœuds

| Type | Domaine | Description | ID |
|------|---------|-------------|-----|
| `folder` | Both | Dossier | `folder::scripts::/path` |
| `script` | Scripts | Script .nvn | `67982` (numérique) |
| `data_file` | Data | Fichier .ion/.csv | `/Clean/Items.ion` |
| `table` | Scripts | Table définie | `67982::table::Items` |
| `function` | Scripts | Fonction définie | `67982::func::StockEvol` |

### Types d'Arêtes

| Type | Description | Exemple |
|------|-------------|---------|
| `contains` | Hiérarchie dossier | folder → script |
| `reads` | Script lit fichier | script → data_file |
| `writes` | Script écrit fichier | script → data_file |
| `imports` | Script importe module | script → script |
| `defines` | Script définit symbole | script → table |
| `sibling` | Même dossier (bidirectionnel) | script ↔ script |

### Direction pour `neighbors`

| Je cherche... | node_id | direction |
|--------------|---------|-----------|
| Scripts qui LISENT un fichier | le fichier | `incoming` |
| Fichiers qu'un script LIT | le script | `outgoing` |
| Scripts dans le même dossier | le script | `siblings` |

---

## 🔌 Usage Programmatique

```python
from env_graph.api import EnvisionGraphAPI

api = EnvisionGraphAPI()

# Construire
api.build()

# Explorer l'arbre
tree = api.get_tree("/", domain="scripts")

# Lire un script
content = api.read("67982")

# Trouver les voisins
neighbors = api.get_neighbors("67982", direction="outgoing", relation_type="reads")

# Recherche
results = api.search("loader", node_types=["script"])

# Grep
matches = api.grep("table Items")
```

---

## 📁 Données

**Input**: `env_scripts/*.nvn` and `mapping.txt` from the `envision/` repo.

**Output** : 
```text
data/network/
├── network.json    # Full graph (nodes + edges)
└── metadata.json   # Generation statistics
```
