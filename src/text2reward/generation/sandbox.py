"""Chargement et exécution protégée d'une fonction de récompense générée par un LLM.

Équivalent de l'étape « Reducing error with code execution » de Text2Reward (§2.2) :
avant tout entraînement, on vérifie que le code généré
    1. respecte les règles du prompt (imports math/numpy seulement, pas de fonctions dangereuses),
    2. se charge et définit compute_reward avec 4 arguments,
    3. s'exécute sans erreur sur des transitions types (avancer, reculer, mur, bord, arrivée),
    4. renvoie un nombre fini, toujours le même pour la même entrée, sans modifier ses entrées,
    5. répond assez vite pour l'entraînement.
Le tout tourne dans un processus séparé avec un timeout (boucle infinie = échec, pas blocage).

Les messages d'erreur sont en anglais : ils seront renvoyés au LLM pour correction (étape suivante).

Usage :
    python -m text2reward.generation.sandbox results/generated/<run_id>
"""

import argparse
import ast
import builtins
import copy
import json
import math
import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path

ALLOWED_MODULES = {"math", "numpy"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "input", "__import__", "breakpoint", "globals"}
DEFAULT_TIMEOUT_S = 20.0
MAX_MS_PER_CALL = 20.0
TIMING_CALLS = 200

# Grille d'exemple au format de docs/interfaces.md (7 x 7). La vraie disposition des murs
# sera celle de gridworld.py ; ici il suffit d'avoir des murs pour tester les collisions.
GRID_SIZE = 7
MAX_STEPS = 40
SAMPLE_WALLS = frozenset({(1, 1), (1, 2), (1, 3), (3, 3), (3, 4), (3, 5), (5, 1), (5, 2)})


class RewardCodeError(Exception):
    """Le code généré ne respecte pas les règles ou ne se charge pas."""


# --------------------------------------------------------------------------- vérification statique

def check_source(source: str) -> list[str]:
    """Analyse le code sans l'exécuter. Retourne la liste des problèmes trouvés."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"SyntaxError at line {error.lineno}: {error.msg}"]

    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    errors.append(f"Forbidden import '{alias.name}' (only math and numpy are allowed)")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module not in ALLOWED_MODULES:
                errors.append(f"Forbidden import from '{node.module}' (only math and numpy are allowed)")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"Forbidden name '{node.id}'")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"Forbidden dunder attribute access '.{node.attr}'")

    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_reward"]
    if not functions:
        errors.append("No top-level function named 'compute_reward'")
    else:
        n_args = len(functions[0].args.args)
        if n_args != 4:
            errors.append(f"compute_reward must take 4 arguments (state, action, next_state, info), got {n_args}")
    return errors


# --------------------------------------------------------------------------- chargement

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict:
    allowed = {name: getattr(builtins, name) for name in dir(builtins) if name not in FORBIDDEN_NAMES}
    allowed["__import__"] = _safe_import
    return allowed


def load_reward_from_source(source: str):
    """Vérifie puis charge le code. Retourne la fonction compute_reward."""
    errors = check_source(source)
    if errors:
        raise RewardCodeError("\n".join(errors))
    namespace = {"__builtins__": _safe_builtins(), "__name__": "generated_reward"}
    exec(compile(source, "<generated_reward>", "exec"), namespace)
    function = namespace.get("compute_reward")
    if not callable(function):
        raise RewardCodeError("'compute_reward' is not callable after loading")
    return function


def load_reward(path):
    """Charge compute_reward depuis un dossier de run ou un fichier reward.py."""
    path = Path(path)
    if path.is_dir():
        path = path / "reward.py"
    return load_reward_from_source(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- transitions types

def _state(agent, goal, step):
    return {
        "agent_pos": agent,
        "goal_pos": goal,
        "walls": SAMPLE_WALLS,
        "grid_size": GRID_SIZE,
        "step": step,
        "max_steps": MAX_STEPS,
    }


def _transition(agent, action, next_agent, goal, collision, step=0, episode_collisions=0):
    reached = next_agent == goal
    info = {
        "collision": collision,
        "reached_goal": reached,
        "is_success": reached,
        "episode_collisions": episode_collisions + int(collision),
    }
    return _state(agent, goal, step), action, _state(next_agent, goal, step + 1), info


def build_test_transitions() -> dict:
    """Transitions types, cohérentes avec les règles de docs/interfaces.md (0 haut, 1 droite, 2 bas, 3 gauche)."""
    goal = (0, 6)
    return {
        "move_closer": _transition((0, 0), 1, (0, 1), goal, collision=False),
        "move_away": _transition((0, 1), 3, (0, 0), goal, collision=False),
        "hit_wall": _transition((0, 2), 2, (0, 2), goal, collision=True),
        "hit_border": _transition((0, 0), 0, (0, 0), goal, collision=True),
        "reach_goal": _transition((0, 5), 1, (0, 6), goal, collision=False),
        "last_step": _transition((0, 0), 1, (0, 1), goal, collision=False, step=MAX_STEPS - 1),
    }


# --------------------------------------------------------------------------- exécution des vérifications

def _short_traceback(max_lines: int = 6) -> str:
    lines = traceback.format_exc().strip().splitlines()
    return "\n".join(lines[-max_lines:])


def run_checks(source: str) -> dict:
    """Charge le code et l'exécute sur les transitions types. Retourne un rapport."""
    report = {"ok": False, "errors": [], "values": {}, "ms_per_call": None}
    try:
        function = load_reward_from_source(source)
    except RewardCodeError as error:
        report["errors"] = str(error).splitlines()
        return report
    except Exception:
        report["errors"] = [f"Error while loading the code:\n{_short_traceback()}"]
        return report

    transitions = build_test_transitions()
    for name, (state, action, next_state, info) in transitions.items():
        snapshot = copy.deepcopy((state, next_state, info))
        try:
            first = function(state, action, next_state, info)
            second = function(state, action, next_state, info)
        except Exception:
            report["errors"].append(f"[{name}] exception raised:\n{_short_traceback()}")
            continue
        if (state, next_state, info) != snapshot:
            report["errors"].append(f"[{name}] the function modified its inputs (it must be pure)")
        try:
            value, value_again = float(first), float(second)
        except (TypeError, ValueError):
            report["errors"].append(f"[{name}] returned {type(first).__name__}, not a number")
            continue
        if not math.isfinite(value):
            report["errors"].append(f"[{name}] returned a non-finite value: {value}")
            continue
        if value != value_again:
            report["errors"].append(f"[{name}] not deterministic: {value} then {value_again}")
        report["values"][name] = value

    if not report["errors"]:
        state, action, next_state, info = transitions["move_closer"]
        start = time.perf_counter()
        for _ in range(TIMING_CALLS):
            function(state, action, next_state, info)
        ms = 1000 * (time.perf_counter() - start) / TIMING_CALLS
        report["ms_per_call"] = round(ms, 4)
        if ms > MAX_MS_PER_CALL:
            report["errors"].append(f"Too slow: {ms:.2f} ms per call (limit {MAX_MS_PER_CALL} ms)")

    report["ok"] = not report["errors"]
    return report


def _worker(source: str, results) -> None:
    try:
        results.put(run_checks(source))
    except BaseException as error:  # dernier filet de sécurité dans le processus fils
        results.put({"ok": False, "errors": [f"{type(error).__name__}: {error}"], "values": {}, "ms_per_call": None})


def validate_source(source: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Lance run_checks dans un processus séparé, avec un timeout."""
    context = mp.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=_worker, args=(source, results))
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"ok": False, "errors": [f"Timeout: no answer after {timeout_s:.0f} s (infinite loop?)"],
                "values": {}, "ms_per_call": None}
    try:
        return results.get(timeout=2)
    except queue.Empty:
        return {"ok": False, "errors": [f"The checking process crashed (exit code {process.exitcode})"],
                "values": {}, "ms_per_call": None}


def validate_run(run_dir) -> dict:
    """Valide results/generated/<run_id>/reward.py et écrit validation.json dans le dossier du run."""
    run_dir = Path(run_dir)
    reward_file = run_dir / "reward.py"
    if not reward_file.exists():
        report = {"ok": False, "errors": ["reward.py not found (code extraction failed)"],
                  "values": {}, "ms_per_call": None}
    else:
        report = validate_source(reward_file.read_text(encoding="utf-8"))
    (run_dir / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Valide une récompense générée.")
    parser.add_argument("run_dir", help="Dossier results/generated/<run_id>")
    args = parser.parse_args()

    report = validate_run(args.run_dir)
    print("OK" if report["ok"] else "ÉCHEC")
    for name, value in report["values"].items():
        print(f"  {name:<12} -> {value:+.4f}")
    if report["ms_per_call"] is not None:
        print(f"  temps par appel : {report['ms_per_call']} ms")
    for error in report["errors"]:
        print(f"  ERREUR : {error}")


if __name__ == "__main__":
    main()