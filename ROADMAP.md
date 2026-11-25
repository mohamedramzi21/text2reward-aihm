# Roadmap du projet

Durée indicative : **6 semaines**, pour une personne ou un petit groupe. Chaque phase produit un résultat démontrable.

| Semaine | Phase | Travail | Livrable / critère de sortie |
| --- | --- | --- | --- |
| 1 | Cadrage et lecture | Résumer la méthode de l’article; choisir GridWorld et les objectifs; fixer les métriques et les baselines. | Cahier des charges court, scénarios de démonstration, protocole expérimental. |
| 2 | Environnement | Implémenter l’environnement 2D et ses observations, actions, obstacles, cible et condition de fin. | Environnement jouable par une politique aléatoire et documenté. |
| 3 | Récompenses | Définir la spécification JSON, les composantes autorisées, leurs bornes et le calcul déterministe de la récompense. Ajouter des exemples validés. | Récompenses clairsemée, manuelle et structurée calculables sur chaque transition. |
| 4 | Génération textuelle | Créer le prompt et le parseur; commencer avec un générateur simulé; ajouter ensuite un adaptateur LLM facultatif et les erreurs gérées. | Consigne → spécification valide, avec affichage de l’explication et des poids. |
| 5 | Entraînement et expérience | Entraîner PPO avec les trois variantes de récompense; répéter avec plusieurs graines; enregistrer les métriques. | Tableau et graphiques comparant réussite, longueur d’épisode et progression. |
| 6 | Démo et rapport | Ajouter une interface de démonstration; analyser les échecs et le reward hacking; documenter les limites et préparer la soutenance. | Démo reproductible, README d’installation, rapport et courte présentation. |

## Critères de réussite du projet

1. Une consigne prise en charge produit une spécification valide et lisible.
2. Une sortie invalide ou hors périmètre est refusée avant l’entraînement.
3. Le même état et la même spécification produisent toujours la même récompense.
4. L’évaluation compare les trois types de récompense sur plusieurs graines et rapporte le taux de réussite, pas seulement la récompense cumulée.
5. La démonstration fonctionne sans clé API grâce au générateur simulé.

## Ordre de priorité

### Indispensable

- Environnement unique et simple.
- Schéma fermé de récompense et validation stricte.
- Générateur simulé reproductible.
- Une méthode RL, trois conditions de comparaison, métriques d’évaluation.
- Documentation et limites expérimentales.

### Si le temps le permet

- Fournisseur LLM réel configurable.
- Retour utilisateur pour réviser la spécification.
- Environnement supplémentaire ou récupération d’exemples proches.
- Détection de comportements indésirables et analyse du reward hacking.

## Protocole expérimental minimal

- **A — Sparse** : récompense seulement lorsque l’agent atteint la cible.
- **B — Dense manuelle** : progression vers la cible, pénalité d’obstacle et coût par étape avec poids fixés par l’équipe.
- **C — Dense générée** : mêmes composantes autorisées, poids et choix suggérés à partir de la consigne.
- Utiliser les mêmes limites d’entraînement et plusieurs graines pour A, B et C.
- Mesurer le taux de réussite sur des épisodes séparés, le nombre moyen d’étapes jusqu’à la cible, la récompense cumulée et le temps ou nombre d’interactions d’entraînement.
- Inspecter des trajectoires pour repérer les solutions qui maximisent le score sans satisfaire réellement l’objectif.

Une différence de score seule ne suffit pas à conclure que la génération améliore l’apprentissage : les résultats doivent être présentés avec les graines, les réglages et les limites.
