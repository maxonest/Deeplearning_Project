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
import threading
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
LOG_DIR = ROOT_DIR / "logs"
BACKEND_LOG_PATH = LOG_DIR / "backend_startup.log"

WINDOWS_EXCEPTION_CODES = {
    0xC0000005: "access violation",
    0xC0000017: "out of memory",
    0xC00000FD: "stack overflow",
    0xC0000135: "missing DLL",
    0xC0000409: "stack buffer overrun / fast fail",
}


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


def stream_process_output(process: subprocess.Popen, log_path: Path) -> None:
    assert process.stdout is not None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()


def start_process(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path | None = None,
) -> subprocess.Popen:
    print(f"Starting: {' '.join(command)}")
    if log_path is None:
        return subprocess.Popen(command, cwd=str(cwd), env=env)

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output_thread = threading.Thread(
        target=stream_process_output,
        args=(process, log_path),
        daemon=True,
    )
    output_thread.start()
    setattr(process, "output_thread", output_thread)
    return process


def describe_return_code(return_code: int) -> str:
    unsigned_code = return_code & 0xFFFFFFFF
    exception_name = WINDOWS_EXCEPTION_CODES.get(unsigned_code)
    if exception_name is None:
        return str(return_code)
    return f"{return_code} (0x{unsigned_code:08X}: Windows {exception_name})"


def main() -> None:
    args = parse_args()
    load_dotenv_if_available()
    backend_port = os.environ.get("BACKEND_PORT", "8000")
    frontend_port = os.environ.get("FRONTEND_PORT", "5173")

    env = os.environ.copy()
    env.setdefault("VITE_API_BASE_URL", f"http://localhost:{backend_port}")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

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
            "-X",
            "faulthandler",
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
        log_path=BACKEND_LOG_PATH,
    )
    print(f"Backend:  http://localhost:{backend_port}")
    print(f"Backend log: {BACKEND_LOG_PATH}")
    print("Backend output is written to the log file and is not shown in this terminal.")
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
                    output_thread = getattr(process, "output_thread", None)
                    if output_thread is not None:
                        output_thread.join(timeout=2)
                    code_description = describe_return_code(return_code)
                    log_hint = (
                        f" See backend log: {BACKEND_LOG_PATH}"
                        if process is backend
                        else ""
                    )
                    raise RuntimeError(
                        f"Service exited with code {code_description}.{log_hint}"
                    )
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping services...")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    main()
