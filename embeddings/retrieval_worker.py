"""Run FAISS and sentence-transformers in an isolated subprocess.

The worker communicates with the FastAPI process using one JSON object per
line on stdin/stdout. Keeping native retrieval libraries outside the model
process prevents a FAISS access violation from terminating the API server.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.embed_utils import FaissKnowledgeBase  # noqa: E402


PROTOCOL_STDOUT = sys.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated local retrieval worker.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--query-prompt-name", default="query")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--faiss-threads", type=int, default=1)
    return parser.parse_args()


def write_response(payload: dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTOCOL_STDOUT.flush()


def main() -> None:
    args = parse_args()
    knowledge_base = FaissKnowledgeBase(
        index_dir=args.index,
        embedding_model=args.model,
        query_prompt_name=args.query_prompt_name or None,
        device=args.device,
        batch_size=args.batch_size,
        faiss_threads=args.faiss_threads,
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            with redirect_stdout(sys.stderr):
                if command == "search":
                    result = knowledge_base.search(
                        str(request.get("query", "")),
                        top_k=int(request.get("top_k", 5)),
                    )
                elif command == "load":
                    knowledge_base.load()
                    result = knowledge_base.index_info()
                elif command == "shutdown":
                    write_response({"ok": True, "result": None})
                    return
                else:
                    raise ValueError(f"Unknown retrieval worker command: {command!r}")
            write_response({"ok": True, "result": result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            write_response(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


if __name__ == "__main__":
    main()
