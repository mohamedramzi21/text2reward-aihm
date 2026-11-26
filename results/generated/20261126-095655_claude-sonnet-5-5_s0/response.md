## What the task means

The agent must reach the goal in as few steps as possible and must not bump into walls or the border. That gives four reward components:

1. **Progress**: reward moves that bring the agent closer to the goal. Walls can make Manhattan distance misleading, for example when a wall sits between the agent and the goal. So I use the true shortest-path distance, computed with a BFS from the goal over free cells. The shaping term is `dist(s) - dist(s')`. This is potential-based (Φ = −dist), so it guides the agent without changing which policies are optimal.
2. **Goal bonus**: a large positive reward (+10) when the goal is reached.
3. **Collision penalty**: −1 when `info["collision"]` is true, since the task says not to bump into walls.
4. **Step cost**: a small −0.05 per step, so shorter paths score higher.

Moving toward the goal nets about +0.45, moving away about −0.55, and a collision about −1.05. Collisions are therefore clearly worse than a harmless detour.

```python
import math
import numpy as np


def compute_reward(state: dict, action: int, next_state: dict, info: dict) -> float:
    # Reward weights
    GOAL_BONUS = 10.0        # reaching the goal
    COLLISION_PENALTY = 1.0  # bumping into a wall / border
    STEP_COST = 0.05         # encourages short paths
    PROGRESS_WEIGHT = 0.5    # scale of the shaping term

    n = int(state["grid_size"])
    walls = state["walls"]
    goal = tuple(next_state["goal_pos"])

    # Shortest-path distance map from the goal, via BFS over free cells
    # (walls respected). Uses only lists, no extra imports.
    dist_map = {goal: 0}
    frontier = [goal]
    while frontier:
        nxt = []
        for (r, c) in frontier:
            d = dist_map[(r, c)]
            for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n \
                        and (nr, nc) not in walls and (nr, nc) not in dist_map:
                    dist_map[(nr, nc)] = d + 1
                    nxt.append((nr, nc))
        frontier = nxt

    def dist(pos):
        pos = tuple(pos)
        if pos in dist_map:
            return float(dist_map[pos])
        # Fallback (unreachable cell): Manhattan distance
        return float(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))

    d_before = dist(state["agent_pos"])
    d_after = dist(next_state["agent_pos"])

    # 1. Progress: potential-based shaping (+1 closer, -1 farther, 0 if stuck)
    reward = PROGRESS_WEIGHT * (d_before - d_after)

    # 2. Goal bonus
    if info.get("reached_goal", False):
        reward += GOAL_BONUS

    # 3. Collision penalty
    if info.get("collision", False):
        reward -= COLLISION_PENALTY

    # 4. Per-step cost
    reward -= STEP_COST

    reward = float(reward)
    if not math.isfinite(reward):
        return 0.0
    return reward
```