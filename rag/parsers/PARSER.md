# 📄 Parsers - Analyseurs de Code

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Envision](https://img.shields.io/badge/Envision-DSL-orange.svg)](https://lokad.com)
[![Regex](https://img.shields.io/badge/Regex-Pattern_Matching-blue.svg)](https://docs.python.org/3/library/re.html)

> *Analyseurs spécialisés pour différents langages de programmation et DSL, avec support avancé pour Envision*

---

## 📁 Contenu du dossier

Le dossier `rag/parsers` contient les analyseurs de code pour différents langages et DSL :

### 📄 Fichiers principaux

- **`__init__.py`** - Module d'initialisation exportant `EnvisionParser`
- **`envision_parser.py`** - Parser moderne pour scripts Envision (.nvn)
- **`old_envision_parser.py`** - Version précédente du parser Envision (legacy)

---

## 🎯 Types de parsers

### 1. 🧠 EnvisionParser (Moderne)

**Parser spécialisé Envision** - Analyse sémantique des scripts DSL Lokad

#### ✨ Fonctionnalités

- 📝 **Extraction de blocs sémantiques** - Identification automatique des structures logiques
- 🔗 **Analyse de dépendances** - Détection des références entre tables et variables
- 📊 **Support complet Envision** - read, write, table, show, const, export, etc.
- 🎯 **Filtrage intelligent** - Exclusion des mots-clés et propriétés communes
- 🔍 **Extraction de contextes** - Analyse des environnements d'utilisation

#### 📋 Types de blocs reconnus

| Type de bloc | Description | Exemple |
|-------------|-------------|---------|
| **📖 Comment** | Commentaires et documentation | `/// Commentaire` |
| **📥 Import** | Importation de modules | `import "module"` |
| **📊 Read** | Lecture de données | `read "/path/file.csv"` |
| **📤 Write** | Écriture de données | `write "/path/output.csv"` |
| **🔧 Const** | Constantes | `const PI = 3.14159` |
| **📤 Export** | Exportation | `export Items` |
| **📋 Table** | Définitions de table | `table Items = ...` |
| **📈 Show** | Visualisations | `show table "Title"` |
| **🎯 Assignment** | Assignations | `Items.Total = sum(Items.Amount)` |

#### 💻 Utilisation

```python
from rag.parsers import EnvisionParser

# Initialisation
parser = EnvisionParser()

# Parsing d'un fichier
blocks = parser.parse_file("script.nvn")

# Parsing de contenu
content = """
/// Section principale
read "/data/input.csv" as Items
Items.Total = sum(Items.Amount)
show table "Résultats" with Items.Total
"""

blocks = parser.parse_content(content, "script.nvn")

# Analyse des résultats
for block in blocks:
    print(f"Bloc: {block.block_type}")
    print(f"Lignes: {block.line_start}-{block.line_end}")
    print(f"Dépendances: {block.dependencies}")
    print(f"Définitions: {block.definitions}")
```

---

### 2. 🏛️ EnvisionParser (Legacy)

**Parser Envision historique** - Version précédente avec configuration avancée

#### ✨ Fonctionnalités

- ⚙️ **Configuration flexible** - Paramètres regex configurables
- 📑 **Sections hiérarchiques** - Analyse par sections délimitées
- 🔤 **Patterns personnalisables** - Expressions régulières adaptables
- 📏 **Validation de limites** - Respect des frontières de sections
- 🔄 **Migration supportée** - Compatible avec l'ancienne architecture

#### ⚙️ Configuration avancée

```python
config = {
    'case_sensitive': False,              # Sensibilité à la casse
    'multiline_patterns': True,           # Patterns multilignes
    'supported_extensions': ['.nvn'],     # Extensions supportées
    'section_delimiter': {
        'min_chars': 20,                  # Longueur minimale délimiteur
        'valid_chars': ['~', '=', '-'],   # Caractères valides
        'pattern_prefix': '///'           # Préfixe des délimiteurs
    }
}
```

#### 📋 Structure de sections

```envision
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Section d'importation des données
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

read "/data/products.csv" as Products
read "/data/sales.csv" as Sales

/// ================================
// Calculs et transformations
/// ================================

Products.TotalValue = Products.Quantity * Products.UnitPrice
Sales.Profit = Sales.Revenue - Sales.Cost

/// -------------------------------
// Visualisations finales
/// -------------------------------

show table "Analyse Produits" with Products.TotalValue
show linechart "Evolution Ventes" with Sales.Profit
```

---

## 🏗️ Architecture commune

Tous les parsers héritent de `BaseParser` et implémentent l'interface unifiée :

### 🔧 Méthodes principales

- `parse_file(file_path)` - Analyse d'un fichier complet
- `parse_content(content, file_path)` - Analyse de contenu texte
- `supported_extensions` - Extensions de fichiers supportées

### 📦 Structure des blocs

Chaque `CodeBlock` contient :

```python
class CodeBlock:
    content: str          # Contenu du bloc
    block_type: BlockType # Type sémantique
    line_start: int       # Ligne de début
    line_end: int         # Ligne de fin
    file_path: str        # Chemin du fichier source
    dependencies: Set[str] # Dépendances détectées
    definitions: Set[str] # Définitions créées
    metadata: Dict        # Métadonnées additionnelles
```

---

## 🔄 Intégration dans le pipeline

Les parsers s'intègrent parfaitement dans le pipeline RAG :

### 📋 Workflow typique

1. **📂 Lecture** - Chargement des fichiers source (.nvn, .py, etc.)
2. **🔍 Analyse** - Extraction des blocs sémantiques
3. **🏷️ Classification** - Attribution de types et métadonnées
4. **🔗 Dépendances** - Analyse des relations entre blocs
5. **📤 Chunking** - Découpage pour l'indexation

### 🎯 Choix du parser

| Langage/DSL | Parser recommandé | Avantages |
|-------------|-------------------|-----------|
| **Envision (.nvn)** | `EnvisionParser` | 📊 Analyse sémantique spécialisée |
| **Python (.py)** | Extension future | 🔧 Support générique |
| **JavaScript** | Extension future | 🌐 Analyse moderne |
| **SQL** | Extension future | 🗃️ Requêtes complexes |

---

## 📦 Dépendances

### 🔧 Bibliothèques externes

- **`re`** - Expressions régulières Python (parsing patterns)
- **`typing`** - Annotations de types avancées
- **`pathlib`** - Gestion moderne des chemins

### 🔗 Modules internes

- `rag.core.base_parser` - Classe de base abstraite
- `config_manager` - Configuration centralisée
- `rag.core.base_chunker` - Interface avec le chunking

---

## 🔧 Extension pour nouveaux langages

Pour ajouter un nouveau parser :

```python
from rag.core.base_parser import BaseParser, CodeBlock, BlockType

class NewLanguageParser(BaseParser):
    @property
    def supported_extensions(self) -> List[str]:
        return ['.ext']
    
    def parse_content(self, content: str, file_path: str = "") -> List[CodeBlock]:
        # Implémentation personnalisée
        blocks = []
        # ... logique de parsing ...
        return blocks
```

Puis ajouter au `__init__.py` :
```python
from rag.parsers.new_parser import NewLanguageParser
__all__ = ["EnvisionParser", "NewLanguageParser"]
```