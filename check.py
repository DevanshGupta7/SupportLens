"""One-command health check (works on Windows, macOS and Linux):  python check.py

Runs the unit tests, trains a fresh model, then runs both agents in offline mode.
"""

import os
import shutil
import subprocess
import sys

PY = sys.executable  # the same Python you used to start this script
OUT = "artifacts_check"


def run(title: str, args: list[str], env_extra: dict | None = None) -> None:
    print(f"\n== {title} ==", flush=True)
    env = {**os.environ, **(env_extra or {})}
    if subprocess.run([PY, *args], env=env).returncode != 0:
        sys.exit(f"\nFAILED at step: {title}")


try:
    run("1/4 unit tests", ["-m", "pytest", "-q"])
    run(
        "2/4 train on synthetic data",
        ["-m", "supportlens.train", "--n", "1500", "--epochs", "6", "--out", OUT],
    )
    point = {"SUPPORTLENS_ARTIFACTS": OUT}
    run(
        "3/4 LangChain agent (offline)",
        ["-m", "supportlens.agent", "my parcel is stuck at the courier hub"],
        point,
    )
    run(
        "4/4 CrewAI agent (offline fallback)",
        ["-m", "supportlens.crew_agent", "my parcel is stuck at the courier hub"],
        point,
    )
finally:
    shutil.rmtree(OUT, ignore_errors=True)
print("\nALL CHECKS PASSED")
