# Text2Reward – génération de récompenses pour l’apprentissage par renforcement

> Projet académique inspiré de « Text2Reward: Reward Shaping with Language Models for Reinforcement Learning » (ICLR 2024).

## Idée

L’utilisateur décrit en langage naturel le comportement souhaité pour un agent. Le système transforme cette consigne en une récompense dense, vérifiable et interprétable, puis entraîne un agent de reinforcement learning dans un environnement simulé.

Exemple : « atteindre la case verte en évitant les obstacles ». Le générateur propose des composantes de récompense telles que la progression vers la cible, une pénalité de collision et un coût par étape. Ces composantes sont calculées à partir de l’état explicite de l’environnement.

## Objectif du projet

Construire et évaluer un prototype pédagogique qui montre comment une consigne textuelle peut guider la conception d’une fonction de récompense et l’apprentissage d’une politique.

### Périmètre du MVP

- Un environnement 2D déterministe de type GridWorld, avec cible, obstacles et état observable.
- Une interface légère pour saisir l’objectif et visualiser la récompense proposée.
- Un générateur interchangeable : mode simulé pour les démonstrations reproductibles, puis fournisseur LLM configurable.
- Une représentation structurée des récompenses, avec opérations autorisées et paramètres validés.
- Un entraînement PPO et une comparaison avec une récompense clairsemée et une récompense dense conçue à la main.
- Des résultats reproductibles : taux de réussite, longueur des épisodes et courbe d’apprentissage.

Le MVP ne reproduit pas les expériences robotiques de l’article. Il en reprend le pipeline central à une échelle adaptée à un projet académique.

## Architecture envisagée

```text
Consigne en langage naturel
            │
            ▼
Générateur de spécification ──► Validation des champs et des poids
            │                                      │
            ▼                                      ▼
Environnement GridWorld ◄── Calculateur de récompense structurée
            │
            ▼
Entraînement PPO ──► Évaluation et comparaison des résultats
```

Le générateur retourne une spécification JSON versionnée plutôt que du Python arbitraire. Un calculateur contrôlé interprète une liste fermée de composantes de récompense. Cela rend l’exécution plus prévisible et permet de vérifier la sortie du modèle avant l’entraînement.

## Installation et lancement

Les commandes seront renseignées avec la première version exécutable. La configuration du fournisseur LLM utilisera des variables d’environnement; aucune clé API ne devra être enregistrée dans Git.

## Feuille de route

Voir [`ROADMAP.md`](ROADMAP.md) pour les phases, livrables et critères de réussite.

## Référence

Tianbao Xie et al., [Text2Reward: Reward Shaping with Language Models for Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2024/file/9941833e8327910ef25daeb9005e4748-Paper-Conference.pdf), ICLR 2024.

## Licence

À choisir avant publication du dépôt (MIT est adaptée à un prototype pédagogique si les dépendances et contributions le permettent).
