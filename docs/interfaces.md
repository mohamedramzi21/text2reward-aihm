# Interfaces communes

## 1. Environnement — `src/text2reward/environment/gridworld.py`

Classe `GridWorldEnv(gymnasium.Env)` :

```python
env = GridWorldEnv(reward_fn=compute_reward, split="train", max_steps=40)
```

| Élément | Définition |
|---|---|
| Grille | 7 × 7, murs à position fixe (disposition définie dans `gridworld.py`) |
| Actions | `Discrete(4)` : 0 = haut, 1 = droite, 2 = bas, 3 = gauche |
| Observation (politique) | `Box(0, 1, shape=(4,), float32)` : `[agent_ligne, agent_col, cible_ligne, cible_col] / 6` |
| Collision | déplacement vers un mur ou hors de la grille : l'agent ne bouge pas, `collision = True` |
| Fin d'épisode | `terminated = True` quand l'agent atteint la cible ; `truncated = True` après 40 pas |
| Départ et cible | une paire (départ, cible) de cases libres distinctes, tirée au hasard |
| Split | l'ensemble des paires est découpé une fois pour toutes (graine fixe) : 80 % `train`, 20 % `test`. L'entraînement n'utilise que `train`, l'évaluation que `test`. |

## 2. État transmis à la récompense

`state` (avant le pas) et `next_state` (après le pas) sont des `dict` :

| Clé | Type | Sens |
|---|---|---|
| `agent_pos` | `tuple[int, int]` | (ligne, colonne) de l'agent |
| `goal_pos` | `tuple[int, int]` | (ligne, colonne) de la cible |
| `walls` | `frozenset[tuple[int, int]]` | cases occupées par un mur |
| `grid_size` | `int` | 7 |
| `step` | `int` | nombre de pas déjà effectués |
| `max_steps` | `int` | 40 |

`info` (la transition) :

| Clé | Type | Sens |
|---|---|---|
| `collision` | `bool` | le déplacement a heurté un mur ou le bord |
| `reached_goal` | `bool` | l'agent est sur la cible après le pas |

## 3. Signature de toute fonction de récompense

```python
def compute_reward(state: dict, action: int, next_state: dict, info: dict) -> float:
    ...
```

- Fonction pure : la même entrée donne toujours la même sortie, sans état global ni aléa.
- Imports autorisés : `math` et `numpy` uniquement.
- Retourne un `float` fini.

| Condition | Emplacement |
|---|---|
| A — sparse | `src/text2reward/reward/baselines.py` → `sparse_reward` |
| B — dense manuelle | `src/text2reward/reward/baselines.py` → `dense_manual_reward` |
| C — générée par LLM | `results/generated/<run_id>/reward.py` → `compute_reward`, commité avec `prompt.txt` et la réponse brute du LLM |

## 4. Métriques (indépendantes de la récompense)

Le `info` renvoyé par `env.step()` contient aussi :
- `is_success` (`bool`) : la cible est atteinte ;
- `episode_collisions` (`int`) : nombre de collisions depuis le début de l'épisode.

Métriques mesurées sur `test`, pour chaque graine :
- taux de réussite ;
- nombre moyen de pas sur les épisodes réussis ;
- nombre moyen de collisions par épisode ;
- temps CPU d'entraînement.

La récompense cumulée ne sert **jamais** à comparer A, B et C, car leurs échelles sont différentes.

## 5. Protocole d'entraînement

- PPO (stable-baselines3), mêmes hyperparamètres pour A, B et C, regroupés dans un seul fichier sous `configs/`.
- Même budget de pas d'entraînement pour toutes les conditions (à fixer après un premier essai).
- 5 graines : 0, 1, 2, 3, 4.