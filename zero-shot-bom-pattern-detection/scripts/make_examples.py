from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for sub in ["examples/patterns", "examples/drawings", "examples/outputs"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    print("Example folders are ready.")


if __name__ == "__main__":
    main()
