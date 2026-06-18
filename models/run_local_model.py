"""Run a local model with Python before starting the web app.

Windows example:
    python models/run_local_model.py --model_path models/qwen/Qwen3.5-9B --prompt "你好，介绍一下你自己"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm import TransformersLLMClient  # noqa: E402
from utils.config import settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local model inference.")
    parser.add_argument("--model_path", default=str(settings.local_model_path))
    parser.add_argument(
        "--lora_adapter_path",
        default=str(settings.local_lora_adapter_path) if settings.local_lora_adapter_path else None,
        help="Optional trained PEFT LoRA adapter directory.",
    )
    parser.add_argument("--prompt", default="你好，请用一句话说明你是谁。")
    parser.add_argument("--max_new_tokens", type=int, default=settings.local_model_max_new_tokens)
    parser.add_argument("--temperature", type=float, default=settings.local_model_temperature)
    parser.add_argument("--top_p", type=float, default=settings.local_model_top_p)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--allow_remote_files", action="store_true", help="Allow downloading files from Hugging Face.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    lora_adapter_path = Path(args.lora_adapter_path) if args.lora_adapter_path else None
    if lora_adapter_path is not None and not lora_adapter_path.is_absolute():
        lora_adapter_path = PROJECT_ROOT / lora_adapter_path
    client = TransformersLLMClient(
        model_path=model_path,
        lora_adapter_path=lora_adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        local_files_only=not args.allow_remote_files,
    )
    print(client.generate(args.prompt, enable_thinking=args.enable_thinking))


if __name__ == "__main__":
    main()
