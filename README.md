# 🤖 Assistant IA pour l'Analyse de Code LOKAD

## 🎯 Objectif du Projet

Ce projet vise à construire un système avancé capable d'assister les *Supply Chain Scientists* de **LOKAD**. L'objectif est de leur permettre de naviguer, comprendre et interroger une base de code complexe écrite en **Envision**, un langage propriétaire de LOKAD, en utilisant des questions en langage naturel.

* **Client** : [LOKAD](https://www.lokad.com)
* **Cadre** : Projet Scientifique Collectif (PSC) - École Polytechnique (X24)

## 💻 Prérequis

Avant de commencer, assurez-vous d'avoir :

* Python 3.10 ou supérieur installé
* pip (gestionnaire de paquets Python)
* Git
* Une clé API OpenAI (pour utiliser GPT-4)
* Une clé API Mistral AI (pour utiliser Mistral)

## 🚀 Installation et Configuration

### 1. Cloner le repository

```bash
git clone https://github.com/ClementLokad/llm-DSL-info-extraction.git
cd llm-DSL-info-extraction
```

### 2. Configuration de l'environnement Python

Créez et activez l'environnement virtuel :

Sur Windows :

```bash
python -m venv env
.\env\Scripts\activate
```

Sur Unix/macOS :

```bash
python -m venv env
source env/bin/activate
```

### 3. Installation des dépendances

Mettez à jour pip et installez les dépendances :

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configuration des API Keys

1. Copiez le fichier de configuration :

```bash
# Sur Windows
copy .env.example .env

# Sur Unix/macOS
cp .env.example .env
```

2. Configurez vos clés API :
   * Créez un compte sur [OpenAI Platform](https://platform.openai.com)
   * Inscrivez-vous sur [Mistral AI](https://mistral.ai)
   * Éditez le fichier `.env` avec vos clés :

```env
OPENAI_API_KEY=votre-clé-api-openai
MISTRAL_API_KEY=votre-clé-api-mistral
```

### 5. Vérification de l'installation

Pour vérifier que tout est correctement installé :

```bash
python -c "from agents import GPTAgent; agent = GPTAgent(); agent.initialize()"
```

Si aucune erreur n'apparaît, votre environnement est prêt !

### Structure du Projet

```
llm-DSL-info-extraction/
├── agents/                 # Package des agents IA
│   ├── __init__.py
│   ├── base.py            # Interface abstraite LLMAgent
│   ├── gpt_agent.py       # Implémentation OpenAI
│   └── mistral_agent.py   # Implémentation Mistral
├── .env.example           # Template de configuration
├── .gitignore            # Fichiers ignorés par git
├── requirements.txt       # Dépendances Python
└── README.md             # Documentation
```

## 💻 Prérequis

Avant de commencer, assurez-vous d'avoir :

* Python 3.10 ou supérieur installé
* pip (gestionnaire de paquets Python)
* Git
* Une clé API OpenAI (pour utiliser GPT-4) (au moins une clé d'API)
* Une clé API Mistral AI (pour utiliser Mistral) (au moins une clé d'API)

## � Installation et Configuration

1. **Cloner le repository**

```bash
git clone https://github.com/ClementLokad/llm-DSL-info-extraction.git
cd llm-DSL-info-extraction
```

2. **Créer un environnement virtuel Python**

```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/macOS
# ou
.\venv\Scripts\activate   # Sur Windows
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement**

```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer .env avec vos clés API
# Vous aurez besoin de :
# - Une clé API OpenAI (https://platform.openai.com)
# - Une clé API Mistral AI (https://mistral.ai)
```

## 🀽�️ Architecture de la Pipeline

Voici l'architecture cible qui sera développée :

### 1. 📚 Flux de Préparation des Connaissances (Hors-ligne)

Ce flux prépare la base de connaissances à partir des scripts Envision bruts.

* **`Data Base`** : L'ensemble des scripts `.nvm` fournis par LOKAD.
* **`Parser`** : Un module intelligent qui analyse les scripts, les segmente en morceaux logiques ("chunks") et en extrait des métadonnées (fonctions, variables, etc.).
* **`RAG (Retrieval Augmented Generation)`** : Une base de données vectorielle (ex: FAISS) qui indexe les chunks et les métadonnées pour une recherche sémantique rapide.

### 2. 💡 Flux d'Exécution et d'Évaluation (En ligne)

Ce flux gère les requêtes de l'utilisateur et génère des réponses validées.

* **`Question`** : La question de l'utilisateur en langage naturel.
* **`Engineered Prompt`** : Un prompt "augmenté" qui combine la question avec les chunks de code les plus pertinents récupérés par le `RAG`.
* **`Main LLM`** : Le modèle de langage principal (ex: GPT-4o) qui génère une réponse à partir du prompt.
* **`Logic Checker`** : Un module de vérification qui contrôle la syntaxe et la cohérence logique de la réponse. Il permet une **boucle de correction** en cas d'erreur.
* **`Final Answer`** : La réponse finale, validée et présentée à l'utilisateur.
* **`Answer Grader (Scorer)`** : Un évaluateur qui note la qualité de la réponse finale en la comparant à une réponse de référence (utilisé pour le benchmark).
