"""Génération d'une fonction de récompense par un LLM (étape zero-shot de Text2Reward).

Chaque génération est sauvegardée dans results/generated/<run_id>/ :
    prompt.txt     le prompt exact envoyé au LLM
    response.md    la réponse brute du LLM (raisonnement + code)
    reward.py      le code extrait (si l'extraction réussit)
    meta.json      modèle, paramètres, tokens, statut de l'extraction

Usage :
    python -m text2reward.generation.generate
    python -m text2reward.generation.generate --n-samples 3
    python -m text2reward.generation.generate --model claude-haiku-5-5 --temperature 0
"""

import argparse
import json
import re
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from text2reward.generation.prompt import DEFAULT_INSTRUCTION, build_zero_shot_prompt

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_DIR = REPO_ROOT / "results" / "generated"
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_TOKENS = 4000

_CODE_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_reward_code(response_text: str) -> str:
    """Retourne le dernier bloc ```python contenant 'def compute_reward'."""
    blocks = [block.strip() for block in _CODE_BLOCK.findall(response_text)]
    candidates = [block for block in blocks if "def compute_reward" in block]
    if not candidates:
        raise ValueError("aucun bloc ```python contenant 'def compute_reward' dans la réponse")
    return candidates[-1] + "\n"


def call_llm(prompt: str, model: str, max_tokens: int, temperature: float | None) -> tuple[str, dict]:
    """Envoie le prompt au LLM et retourne (texte de la réponse, métadonnées d'usage)."""
    client = anthropic.Anthropic()
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.messages.create(**kwargs)
    text = "".join(block.text for block in response.content if block.type == "text")
    usage = {
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, usage


def generate_one(
    instruction: str,
    model: str,
    max_tokens: int,
    temperature: float | None,
    run_id: str,
) -> Path:
    """Fait une génération complète et sauvegarde tous les artefacts. Retourne le dossier du run."""
    prompt = build_zero_shot_prompt(instruction)
    text, usage = call_llm(prompt, model, max_tokens, temperature)

    out_dir = GENERATED_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "response.md").write_text(text, encoding="utf-8")

    meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "instruction": instruction,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "anthropic_sdk": version("anthropic"),
        **usage,
    }
    try:
        code = extract_reward_code(text)
        (out_dir / "reward.py").write_text(code, encoding="utf-8")
        meta["extraction"] = "ok"
    except ValueError as error:
        meta["extraction"] = f"echec: {error}"

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère des fonctions de récompense avec un LLM.")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Par défaut : valeur par défaut de l'API.")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Nombre de générations indépendantes.")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for k in range(args.n_samples):
        run_id = f"{stamp}_{args.model}_s{k}"
        out_dir = generate_one(args.instruction, args.model, args.max_tokens, args.temperature, run_id)
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        print(
            f"[{k + 1}/{args.n_samples}] {run_id} | extraction: {meta['extraction']} | "
            f"stop: {meta['stop_reason']} | tokens: {meta['input_tokens']} in, {meta['output_tokens']} out"
        )


if __name__ == "__main__":
    main()