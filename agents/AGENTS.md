# 🤖 Documentation des Agents IA

[![Agents](https://img.shields.io/badge/Agents-GPT%20%7C%20Gemini%20%7C%20Mistral-blue.svg)](https://github.com)
[![Rate Limiting](https://img.shields.io/badge/Rate_Limiting-Enabled-green.svg)](https://github.com)

> *Architecture modulaire et extensible pour l'intégration de modèles de langage*

---

## 🎯 Vue d'ensemble

Le système d'agents IA fournit une **interface unifiée** pour interagir avec différents modèles de langage tout en gérant automatiquement :

- 🔄 **Rate limiting intelligent** avec retry exponential backoff
- 🔌 **Architecture plugin** pour faciliter l'ajout de nouveaux agents
- ⚡ **Gestion d'erreurs robuste** avec fallback et logging
- 🎛️ **Configuration externalisée** des paramètres par agent
- 📊 **Métriques de performance** et monitoring des appels API

---

## 🏗️ Architecture des agents

```text
┌─────────────────────────────────────────────────────────────┐
│                  🤖 LLM AGENTS ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────┤
│  🎭 Agent Interface (base.py)                              │
│  ├── 🔄 @rate_limited decorator                            │
│  ├── ⚡ Error handling & retries                           │
│  └── 📊 Performance monitoring                             │
├─────────────────────────────────────────────────────────────┤
│  🧠 Agent Implementations                                  │
│  ├── 🔮 GeminiAgent    ├── 🧠 GPTAgent    ├── ⚡ MistralAgent │
├─────────────────────────────────────────────────────────────┤
│  🔧 Configuration Layer                                    │
│  ├── 🔑 API Keys (.env)  ├── ⚙️ Parameters (config.yaml)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Agents disponibles

### 🔮 Gemini Agent - Google AI

**Fichier**: `gemini_agent.py`
**Provider**: Google AI Studio
**Modèle**: `gemini-1.5-flash`

#### ✨ Caractéristiques

- ⚡ **Vitesse exceptionnelle** - Réponses sub-seconde
- 🧠 **Compréhension de code** - Optimisé pour l'analyse syntaxique
- 🌍 **Multilingue** - Support excellent français/anglais
- 💰 **Coût-efficace** - Tarification attractive pour volume
- 🔒 **Sécurité** - Policies de contenu strictes

#### 🔧 Configuration

```yaml
# config.yaml
agent:
  default_model: "gemini"
  
gemini:
  model_name: "gemini-1.5-flash"
  temperature: 0.1          # Précision maximale
  max_output_tokens: 2048   # Réponses détaillées
  top_p: 0.8               # Créativité contrôlée
```

```bash
# .env
GOOGLE_API_KEY=your-gemini-api-key-here
```

#### 📊 Métriques typiques

- **Latence** : ~0.8s par requête
- **Throughput** : 60 requêtes/minute
- **Token limits** : 1M tokens contexte, 8K output
- **Rate limits** : Gérés automatiquement

#### 🎯 Cas d'usage optimaux

- ✅ Analyse rapide de code DSL
- ✅ Extraction d'informations factuelles
- ✅ Questions-réponses sur documentation
- ✅ Résumés et synthèses

---

### 🧠 GPT Agent - OpenAI

**Fichier**: `gpt_agent.py`
**Provider**: OpenAI API
**Modèle**: `gpt-4-turbo`

#### ✨ Caractéristiques

- 🏆 **Qualité premium** - Raisonnement le plus sophistiqué
- 🔍 **Analyse profonde** - Compréhension contextuelle excellente
- 🎨 **Créativité** - Génération de contenu riche
- 📚 **Knowledge base** - Formation sur données récentes
- 🔧 **Function calling** - Support des outils externes

#### 🔧 Configuration

```yaml
# config.yaml
gpt:
  model_name: "gpt-4-turbo"
  temperature: 0.2          # Balance précision/créativité
  max_tokens: 4096         # Réponses étendues
  top_p: 0.9               # Diversité vocabulaire
  frequency_penalty: 0.1   # Éviter répétitions
  presence_penalty: 0.1    # Diversifier sujets
```

```bash
# .env  
OPENAI_API_KEY=your-openai-api-key-here
```

#### 📊 Métriques typiques

- **Latence** : ~2.5s par requête
- **Throughput** : 20 requêtes/minute (Tier 1)
- **Token limits** : 128K contexte, 4K output
- **Rate limits** : Variables selon tier

#### 🎯 Cas d'usage optimaux

- ✅ Raisonnement complexe sur business logic
- ✅ Génération de documentation détaillée
- ✅ Explication de patterns architecturaux
- ✅ Debug et troubleshooting

---

### ⚡ Mistral Agent - Mistral AI

**Fichier**: `mistral_agent.py`
**Provider**: Mistral AI
**Modèle**: `mistral-large-latest`

#### ✨ Caractéristiques

- 🇪🇺 **Alternative européenne** - Conformité RGPD native
- ⚖️ **Balance prix/performance** - Excellent rapport qualité/coût
- 🌍 **Multilingue avancé** - Spécialisé langues européennes
- 🔒 **Privacy-focused** - Données traitées en Europe
- ⚡ **Vitesse compétitive** - Latence raisonnable

#### 🔧 Configuration

```yaml
# config.yaml
mistral:
  model_name: "mistral-large-latest"
  temperature: 0.3          # Équilibre optimal
  max_tokens: 3000         # Réponses substancielles
  top_p: 0.85              # Contrôle qualité
  random_seed: 42          # Reproductibilité tests
```

```bash
# .env
MISTRAL_API_KEY=your-mistral-api-key-here
```

#### 📊 Métriques typiques

- **Latence** : ~1.8s par requête
- **Throughput** : 30 requêtes/minute
- **Token limits** : 32K contexte, 8K output
- **Rate limits** : Généreux pour développement

#### 🎯 Cas d'usage optimaux

- ✅ Projets nécessitant conformité européenne
- ✅ Applications multilingues français/anglais
- ✅ Budget contraint avec qualité maintenue
- ✅ Prototypage et développement

---

## 🔧 Interface abstraite - BaseAgent

### 📋 Classe de base

```python
# agents/base.py
from abc import ABC, abstractmethod
from typing import Optional
import time
import functools

class LLMAgent(ABC):
    """Interface abstraite pour tous les agents IA"""
  
    @abstractmethod
    def initialize(self) -> None:
        """Initialiser l'agent avec sa configuration"""
        pass
  
    @abstractmethod  
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """Générer une réponse à partir de la question et du contexte
    
        Args:
            question: Question posée par l'utilisateur
            context: Contexte extrait des chunks pertinents
        
        Returns:
            Réponse générée par le modèle
        """
        pass
```

### 🔄 Décorateur Rate Limiting

```python
@rate_limited(max_retries: int = 3, initial_delay: float = 1.0)
def decorator(func):
    """
    Gestion intelligente des limites de taux API:
  
    Features:
    - ⏱️ Retry avec exponential backoff  
    - 🎯 Détection automatique rate limits
    - 📊 Logging détaillé des tentatives
    - ⚡ Pause minimale après succès
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                time.sleep(0.1)  # Pause courtoise
                return result
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = initial_delay * (2 ** attempt)
                    print(f"⏳ Rate limit atteint. Attente {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
        return wrapper
```

---

## 🎛️ Configuration avancée

### ⚙️ Paramètres par défaut

```yaml
# config.yaml - Section agents
agent:
  # Agent par défaut du système
  default_model: "mistral"
```

### 🔑 Gestion des clés API

```bash
# .env - Clés d'API sécurisées
GOOGLE_API_KEY=AIza...                    # Google AI Studio
OPENAI_API_KEY=sk-proj-...               # OpenAI API
MISTRAL_API_KEY=H3I21h...                # Mistral AI

# Variables optionnelles pour configuration avancée
GOOGLE_PROJECT_ID=your-project-id        # Pour Google Cloud
OPENAI_ORG_ID=your-org-id               # Pour organisation OpenAI
```

---

## 🧪 Tests et validation

### 📊 Suite de tests agents

**Fichier**: `test_agents.py`

```python
class TestAgents:
    """Tests exhaustifs des agents IA"""
  
    def test_agent_initialization(self):
        """Vérifier initialisation correcte de tous les agents"""
    
    def test_rate_limiting(self):
        """Valider le comportement du rate limiting"""
    
    def test_error_handling(self):
        """Tester gestion d'erreurs et fallbacks"""
    
    def test_response_quality(self):
        """Évaluer qualité des réponses sur cas tests"""
    
    def test_performance_metrics(self):
        """Mesurer latence et throughput par agent"""
```

### 🎯 Validation qualité réponses

```python
# Métriques automatisées de qualité
def evaluate_response_quality(question: str, response: str, expected_keywords: List[str]) -> float:
    """
    Critères d'évaluation:
    - ✅ Présence mots-clés attendus
    - ✅ Longueur réponse appropriée  
    - ✅ Structure et formatage
    - ✅ Pertinence contextuelle
    - ✅ Absence d'hallucinations
    """
    score = 0.0
  
    # Keyword matching (40%)
    keyword_score = sum(1 for kw in expected_keywords if kw.lower() in response.lower())
    score += (keyword_score / len(expected_keywords)) * 0.4
  
    # Length appropriateness (20%)
    response_length = len(response.split())
    if 50 <= response_length <= 300:  # Sweet spot pour DSL queries
        score += 0.2
    
    # Structure quality (40%)
    structure_indicators = [':', '-', '•', '\n', 'exemple', 'logic']
    structure_score = sum(1 for indicator in structure_indicators if indicator in response.lower())
    score += min(structure_score / len(structure_indicators), 1.0) * 0.4
  
    return min(score, 1.0)
```

---

## 🚀 Ajout d'un nouvel agent

### 📋 Template d'implémentation

```python
# agents/claude_agent.py
import os
from typing import Optional
import requests

from .base import LLMAgent, rate_limited

class ClaudeAgent(LLMAgent):
    """Agent utilisant Anthropic Claude"""
  
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.model = self.config.get('model_name', 'claude-3-sonnet-20240229')
        self.temperature = self.config.get('temperature', 0.2)
        self.max_tokens = self.config.get('max_tokens', 3000)
    
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY non configurée")
  
    def initialize(self) -> None:
        """Initialiser le client Claude"""
        self.client = AnthropicClient(api_key=self.api_key)
    
        # Test de connectivité
        try:
            self._test_connection()
            print(f"   ✅ Claude-{self.model} initialisé")
        except Exception as e:
            raise RuntimeError(f"❌ Échec initialisation Claude: {e}")
  
    @rate_limited(max_retries=3, initial_delay=1.0)
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """Générer réponse via Claude API"""
    
        # Construction du prompt
        prompt = self._build_prompt(question, context)
    
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        
            return response.content[0].text.strip()
        
        except Exception as e:
            raise RuntimeError(f"Erreur Claude API: {e}")
```

### ⚙️ Configuration du nouvel agent

```bash
# .env - Ajouter clé API
ANTHROPIC_API_KEY=your-claude-api-key-here
```

---

## 📊 Monitoring et métriques

### 🎯 Métriques collectées automatiquement

```python
class AgentMetrics:
    """Collecteur de métriques pour agents"""
  
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency': 0.0,
            'rate_limit_hits': 0,
            'tokens_consumed': 0,
            'cost_estimate': 0.0
        }
  
    def record_request(self, agent_name: str, latency: float, 
                      success: bool, tokens: int = 0):
        """Enregistrer métriques d'une requête"""
        self.metrics['total_requests'] += 1
    
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Mise à jour latence moyenne
        self._update_avg_latency(latency)
    
        # Tracking tokens et coût
        self.metrics['tokens_consumed'] += tokens
        self.metrics['cost_estimate'] += self._estimate_cost(agent_name, tokens)
```

### 📈 Dashboard métriques

```bash
# Commande de monitoring
python main.py --status --verbose

# Output exemple:
🤖 AGENT PERFORMANCE METRICS
=====================================
📊 Mistral Agent:
   • Requêtes totales: 147
   • Taux de succès: 98.6%
   • Latence moyenne: 1.83s
   • Tokens consommés: 23,456
   • Coût estimé: $0.47
   
⚡ Rate Limiting:
   • Hits détectés: 3
   • Retry réussis: 3/3
   • Temps attente total: 12.4s
```

---

## ⚠️ Troubleshooting

### 🔧 Problèmes courants

#### ❌ Erreur: Clé API invalide

```bash
# Vérifier variables d'environnement
echo $GOOGLE_API_KEY
echo $OPENAI_API_KEY  
echo $MISTRAL_API_KEY

# Recharger .env si nécessaire
source .env
```

#### ❌ Rate limit dépassé

```python
# Le décorateur @rate_limited gère automatiquement
# Mais vous pouvez ajuster les paramètres:

@rate_limited(max_retries=5, initial_delay=2.0)
def generate_response(self, question: str, context: str = None) -> str:
    # Implementation
```

#### ❌ Timeout de connexion

```yaml
# Augmenter timeout dans config.yaml
agent:
  default_timeout: 60  # 60 secondes au lieu de 30
```

### 🔧 Debug et diagnostics

```bash
# Mode debug pour voir tous les appels API
python main.py --verbose --query "test query"

# Test de connectivité d'un agent spécifique
python test_agents.py --agent gemini

# Vérification configuration
python -c "from config_manager import ConfigManager; print(ConfigManager().get_config())"
```

---

## 🤝 Contribution

### 📋 Guidelines pour nouveaux agents

1. **🏗️ Hériter de LLMAgent** - Utiliser l'interface abstraite
2. **🔄 Implémenter @rate_limited** - Gestion automatique des limits
3. **⚙️ Configuration externalisée** - Paramètres dans config.yaml
4. **🧪 Tests exhaustifs** - Suite de tests pour chaque agent
5. **📖 Documentation** - Ajouter exemples et cas d'usage

### 🔧 Checklist nouvel agent

- [ ] Classe hérite de `LLMAgent`
- [ ] Méthodes `initialize()` et `generate_response()` implémentées
- [ ] Décorateur `@rate_limited` appliqué
- [ ] Configuration dans `config.yaml`
- [ ] Variables d'environnement documentées
- [ ] Tests unitaires ajoutés
- [ ] Documentation mise à jour

---

## 🔗 Related Documentation

### Pipeline & Orchestration
- [LangGraph Pipeline](../pipeline/PIPELINE.md) - Main orchestration system
- [Agentic Workflow](../pipeline/agent_workflow/AGENTIC_WORKFLOW.md) - Agent loop using these agents
- [Benchmarking Framework](../pipeline/benchmarks/BENCHMARKS.md) - Performance evaluation using agents

### RAG Components
- [RAG Pipeline Overview](../rag/RAG.md) - Complete retrieval architecture
- [Query Transformers Documentation](../rag/query_transformers/QUERY_TRANSFORMERS.md) - Query enhancement
- [Embedders Documentation](../rag/embedders/EMBEDDERS.md) - Embedding models
- [Retrievers Documentation](../rag/retrievers/RETRIEVERS.md) - Vector database retrieval

### Getting Started
- [Quick Start Tutorial](../TUTORIAL.md) - Setup and usage guide
- [Project README](../README.md) - High-level overview

---

## 📝 License

This project is under PRIVATE LICENSE AGREEMENT. See [LICENSE](../LICENSE) for details.

---
