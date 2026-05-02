from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "engagement"
        / "baselines"
        / "ml_baseline"
        / "run_ml_baselines.py"
    )
    runpy.run_path(str(target), run_name="__main__")
