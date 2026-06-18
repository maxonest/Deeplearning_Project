"""Start backend and frontend on Windows without Bash.

Run from project root:
    python start_windows.py
    python start_windows.py --backend-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the local QA backend and frontend.")
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Start only FastAPI, without installing or starting the frontend.",
    )
    return parser.parse_args()


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT_DIR / ".env")


def start_process(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen:
    print(f"Starting: {' '.join(command)}")
    return subprocess.Popen(command, cwd=str(cwd), env=env)


def main() -> None:
    args = parse_args()
    load_dotenv_if_available()
    backend_port = os.environ.get("BACKEND_PORT", "8000")
    frontend_port = os.environ.get("FRONTEND_PORT", "5173")

    env = os.environ.copy()
    env.setdefault("VITE_API_BASE_URL", f"http://localhost:{backend_port}")

    npm = None
    if not args.backend_only:
        npm = shutil.which("npm.cmd") or shutil.which("npm")
        if npm is None:
            raise RuntimeError("npm was not found. Install Node.js 18+ and reopen the terminal.")
        if not (FRONTEND_DIR / "node_modules").exists():
            print("Installing frontend dependencies...")
            subprocess.run([npm, "install"], cwd=str(FRONTEND_DIR), env=env, check=True)

    backend = start_process(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            backend_port,
        ],
        cwd=ROOT_DIR,
        env=env,
    )
    print(f"Backend:  http://localhost:{backend_port}")
    processes = [backend]

    if not args.backend_only:
        assert npm is not None
        frontend = start_process(
            [
                npm,
                "run",
                "dev",
                "--",
                "--host",
                "0.0.0.0",
                "--port",
                frontend_port,
            ],
            cwd=FRONTEND_DIR,
            env=env,
        )
        processes.append(frontend)
        print(f"Frontend: http://localhost:{frontend_port}")

    print("Press Ctrl+C to stop the service(s).")

    try:
        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(f"Service exited with code {return_code}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping services...")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    main()
