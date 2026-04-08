# Envision — Projet Scientifique Collectif X24

## Titre du projet

**Architecture agentique et hybride pour l'extraction de connaissances sur des bases de code proprietaires (Envision)**

## Composition du groupe

- Degot-Silvestre Gaetan
- Dorchies Yoan
- Kamdem Ivann
- Thebault Guilhem
- Guediche Adam

## Objectif

Le projet traite d'une limitation pratique des grands modeles de langage (LLM) dans les contextes industriels :
les langages specifiques au domaine (DSL) proprietaires ne figurent pas dans les corpus d'entrainement publics
et provoquent des hallucinations frequentes.

L'objectif est de concevoir et evaluer un assistant robuste capable de repondre a des questions techniques
sur la base de code Envision de Lokad en combinant :

- recherche semantique pour les questions conceptuelles ;
- recherche lexicale exacte pour les identifiants, variables et chemins ;
- orchestration agentique pour iterer jusqu'a collecte suffisante de preuves.

## Approche

Le systeme repose sur une architecture hybride et agentique :

- **Preparation des donnees :** parser et chunking semantique adaptes aux scripts Envision.
- **Recherche hybride :** route combinant recherche vectorielle (conceptuelle) et recherche GREP (exacte) pour reduire faux positifs et faux negatifs.
- **Orchestration agentique :** workflow en boucle pour planification, recherche, synthese et verification.
- **Evaluation :** approche benchmark-first avec metriques de qualite et operationnelles (similarite, LLM-as-a-Judge, latence).

## Resultats

L'implementation finale inclut :

- pile de recherche hybride (recherche vectorielle + recherche par grep) ;
- routage de requetes entre modes de recherche conceptuel et exact ;
- workflow agentique avec raisonnement iteratif et appels d'outils ;
- pipeline de benchmarking et de notation ;
- mode d'interaction live avec memoire persistante et compaction de contexte ;
- execution pilotee par configuration et support multi-modeles.

## Illustration

*Figure 2 — Architecture du flux agentique implemente avec LangGraph*

```
  ┌───────────────────┐
  │ Question          │
  │ Utilisateur       │
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │ Agent             │
  │ Planificateur     │
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │ Besoin d'info ?   │◄─────────────────┐
  └───┬───────────┬───┘                  │
      │           │                      │
   Exact       Concept                   │
      │           │                      │
  ┌───▼───┐   ┌───▼───┐                 │
  │ Outil │   │ Outil │                 │
  │ GREP  │   │  RAG  │                 │
  └───┬───┘   └───┬───┘                 │
      │           │                      │
  ┌───▼───────────▼───┐                 │
  │ Agregation        │                 │
  │ Contexte          │                 │
  └─────────┬─────────┘                 │
            │                            │
  ┌─────────▼─────────┐                 │
  │ Generation        │                 │
  │ Reponse           │                 │
  └─────────┬─────────┘                 │
            │                            │
  ┌─────────▼─────────┐    Insuffisant  │
  │ Verification      ├────────────────►┘
  └─────────┬─────────┘
            │ Suffisant
  ┌─────────▼─────────┐
  │ Reponse Finale    │
  └───────────────────┘
```

## Depot du projet

[github.com/ClementLokad/llm-DSL-info-extraction](https://github.com/ClementLokad/llm-DSL-info-extraction)
