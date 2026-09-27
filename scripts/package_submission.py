"""Create a reviewable assignment ZIP from an explicit source allowlist."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("README.md", "Makefile", "pyproject.toml", ".gitignore", ".gcloudignore")
DIRECTORIES = ("src", "tests", "configs", "infra", "terraform", "scripts", "docs", "dataset", "experiments")
SKIP_DIRS = {".git", ".venv", ".terraform", "__pycache__", ".agents", ".codex", "doc"}
SKIP_FILES = {"backend.hcl", ".env", ".env.local"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in relative.parts):
        return False
    if path.name in SKIP_FILES or path.name.startswith(".env."):
        return False
    if path.suffix in {".pyc", ".sqlite3", ".zip"}:
        return False
    if ".tfstate" in path.name or (path.suffix == ".tfvars" and not path.name.endswith(".example")):
        return False
    return path.is_file() and not path.is_symlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "submission" / "campaign-generator-assignment.zip"
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a previously generated ZIP")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {output}")
    candidates = [ROOT / name for name in FILES]
    for directory in DIRECTORIES:
        candidates.extend((ROOT / directory).rglob("*"))
    files = sorted({path for path in candidates if included(path)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    print(f"{output}: {len(files)} files, {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
