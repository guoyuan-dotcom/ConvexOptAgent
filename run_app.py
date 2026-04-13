from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        from convexopt_tutor_agent.execution.local_runner import WORKER_MODE_ARG, run_worker_mode
        from convexopt_tutor_agent.app import main as app_main
    except ImportError as exc:
        print("Failed to import application dependencies.")
        print("Run: py -m pip install -r requirements.txt")
        print(f"Details: {exc}")
        return 1

    if len(sys.argv) >= 4 and sys.argv[1] == WORKER_MODE_ARG:
        return run_worker_mode(sys.argv[2], sys.argv[3])

    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
