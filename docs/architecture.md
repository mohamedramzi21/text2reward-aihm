# Architecture technique proposée

## Modules

- `environment` : GridWorld, état observable, actions et transitions.
- `reward.schema` : schéma versionné des composantes autorisées et validation des types, bornes et noms.
- `reward.engine` : calcul déterministe de la récompense à partir d’une spécification valide et d’une transition.
- `generation` : interface commune entre générateur simulé et fournisseur LLM.
- `training` : entraînement PPO, sauvegarde de configuration et de checkpoints.
- `evaluation` : évaluation sur des épisodes et graines distincts, export des métriques.
- `ui` : formulaire de consigne, visualisation de la spécification, lancement et comparaison des expériences.

## Contrat de génération

Le modèle doit retourner un objet structuré, par exemple :

```json
{
  "version": 1,
  "goal": "atteindre la cible en évitant les obstacles",
  "components": [
    {"name": "progress_to_goal", "weight": 1.0},
    {"name": "collision_penalty", "weight": -1.0},
    {"name": "step_cost", "weight": -0.02}
  ],
  "explanation": "Favorise la progression, pénalise les collisions et les détours."
}
```

L’exemple illustre le format; les noms et limites définitifs seront fixés après l’implémentation de l’environnement. Le validateur rejette les clés inconnues, les valeurs non finies, les poids hors limites et les composantes indisponibles dans l’environnement. Le calcul de récompense reste dans le code du projet.

## Choix techniques de départ

- Python pour l’environnement, l’entraînement et l’évaluation.
- Gymnasium comme interface d’environnement.
- Stable-Baselines3 avec PPO si compatible avec l’environnement retenu.
- Streamlit pour une interface de démonstration rapide.
- JSON pour la spécification et les résultats échangeables.

Ces choix sont des propositions de départ; vérifier les versions compatibles au moment de créer l’environnement logiciel.

## Structure du dépôt visée

```text
text2reward/
├── README.md
├── ROADMAP.md
├── pyproject.toml
├── .env.example
├── docs/
│   ├── architecture.md
│   └── experiment-protocol.md
├── src/text2reward/
│   ├── environment/
│   ├── reward/
│   ├── generation/
│   ├── training/
│   └── evaluation/
├── app/
├── configs/
└── results/
```

Les sorties lourdes d’entraînement, secrets et checkpoints ne sont pas versionnés. Le projet doit garder des fichiers de configuration, les scripts de reproduction et des résultats agrégés légers.
