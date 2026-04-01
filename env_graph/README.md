# Envision Preprocess (`src/envision_preprocess`)

**Les "Yeux" du système.**

Ce package parse les scripts Envision DSL (`.nvn`) et construit un **Graphe de Dépendances Hiérarchique** à deux domaines : scripts et données.

---

## 🚀 Quick Start

```bash
# 1. Construire le graphe de dépendances
uv run network --build

# 2. Explorer la structure
uv run network --tree

# 3. Voir les statistiques
uv run network --stats
```

---

## 📋 CLI Reference

```
uv run network [OPTIONS]

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

Options:
  --json                Sortie JSON (machine-readable)
  --raw                 Sortie brute (sans formatage)
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
parsing:
  script_dir: "scripts"           # Dossier des scripts .nvn
  script_ext: "nvn"               # Extension
  data_extensions: ["ion"]        # Extensions data files
  normalize_brackets: true        # [ 1 ] normalization

api:
  mode: "lite"                    # "lite" (LLM) ou "full" (debug)

output:
  network_file: "datas/network/network.json"
  metadata_file: "datas/network/metadata.json"
```

### Mode Lite vs Full

| Mode | Description | Économie Tokens |
|------|-------------|-----------------|
| `lite` | Minimal, stats préservés | **-80%** sur `get_tree` |
| `full` | Toutes les métadonnées | Pour debug |

---

## 🏗️ Architecture

```
envision_preprocess/
├── api.py          # API publique (EnvisionGraphAPI)
├── builder.py      # Construction du graphe
├── extractor.py    # Extraction des symboles
├── typedefs.py     # Types (NodeType, EdgeType, etc.)
├── network.py      # CLI entry point
└── config.yaml     # Configuration
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
from envision_preprocess.api import EnvisionGraphAPI

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

**Input** : `scripts/*.nvn` (scripts Envision bruts)

**Output** : 
```
datas/network/
├── network.json    # Graphe complet (nodes + edges)
└── metadata.json   # Statistiques de génération
```

