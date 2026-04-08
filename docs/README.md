# Page publique - PSC X24

## Titre du projet

**Architecture agentique et hybride pour l'extraction de connaissances sur des bases de code proprietaires (Envision)**

## Composition du groupe

- Degot-Silvestre Gaetan
- Dorchies Yoan
- Kamdem Ivann
- Thebault Guilhem
- Guediche Adam

## Positionnement academique

Cette page constitue la page publique de communication de la phase de finalisation du PSC **Envision**.
Elle est conforme aux exigences PSC pour la livraison finale :

- rapport final (30 a 40 pages) ;
- page publique contenant la composition du groupe, le titre du projet et au moins une illustration ;
- soutenance orale devant le jury.

## Perimetre et objectif du projet

Le projet traite d'une limitation pratique des LLM dans les contextes industriels : les DSL proprietaires
ne figurent pas dans les corpus d'entrainement publics et provoquent donc des hallucinations frequentes.

Notre objectif est de concevoir et evaluer un assistant robuste capable de repondre a des questions
techniques sur la base de code Envision de Lokad en combinant :

- recherche semantique pour les questions conceptuelles ;
- recherche lexicale exacte pour les questions portant sur des variables, chemins ou identifiants ;
- orchestration agentique pour iterer jusqu'a collecte suffisante de preuves.

## Systeme actuel (etat de la branche main)

L'implementation a progresse au-dela du rapport intermediaire et inclut desormais :

- pile de recherche hybride (recherche vectorielle + recherche par grep) ;
- routage de requetes entre modes de recherche conceptuel et exact ;
- workflow agentique avec raisonnement iteratif et appels d'outils ;
- pipeline de benchmarking et de notation ;
- mode d'interaction live avec memoire persistante et compaction de contexte ;
- execution pilotee par configuration et support multi-modeles.

## Points cles de la methodologie

- **Preparation des donnees :** parser et chunking semantique adaptes aux scripts Envision.
- **Architecture de recherche :** route hybride pour reduire faux positifs et faux negatifs.
- **Controle agentique :** workflow en boucle pour planification, recherche, synthese et verification.
- **Evaluation :** approche benchmark-first avec metriques de qualite et operationnelles.

## Illustration

```
  ┌───────────────────────┐
  │  Question utilisateur │
  └───────────┬───────────┘
              │
        ┌─────▼─────┐
        │  Routeur   │
        └──┬──────┬──┘
           │      │
  Conceptuel    Exact
           │      │
  ┌────────▼──┐ ┌─▼──────────┐
  │ Recherche │ │ Recherche  │
  │vectorielle│ │    GREP    │
  └────────┬──┘ └─┬──────────┘
           │      │
      ┌────▼──────▼────┐
      │  Orchestrateur  │
      │   agentique     │
      └───────┬─────────┘
              │
  ┌───────────▼───────────┐
  │ Agregation contexte   │
  └───────────┬───────────┘
              │
     ┌────────▼────────┐
     │  Synthese LLM   │
     └────────┬────────┘
              │
  ┌───────────▼───────────┐
  │  Controle qualite     │
  └───────────┬───────────┘
              │
     ┌────────▼────────┐
     │ Reponse finale  │
     └─────────────────┘
```

## Dates cles

- Date limite rapport final : **24/04/2026**
- Periode des soutenances : **mai 2026**
- Soutenance prevue : **19/05/2026 a 15h00** (Europe/Paris)

## Liens du projet

- Depot principal du projet (Clement) : [https://github.com/ClementLokad/llm-DSL-info-extraction](https://github.com/ClementLokad/llm-DSL-info-extraction)
- Page de presentation publiee : [https://kpihx.github.io/envision-copilot-presentation/](https://kpihx.github.io/envision-copilot-presentation/)

## Documents source

- `Livret_PSCX24.pdf`
- `PSC_Rapport_Intermediaire_X24.pdf`
