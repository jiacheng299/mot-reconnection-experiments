"""Fetch upstream dependency trees at the archived commits."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    subprocess.run(["git", *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Dependency names from vendor-lock.json")
    args = parser.parse_args()
    entries = json.loads((ROOT / "vendor-lock.json").read_bytes())
    vendor = ROOT / "vendor"
    vendor.mkdir(exist_ok=True)
    for name in args.names or entries:
        entry = entries[name]
        path = vendor / name
        if not path.exists():
            git("clone", entry["url"], path)
        if not (path / ".git").exists():
            raise SystemExit(f"Existing directory is not a dependency clone: {path}")
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"],
                                        text=True).strip()
        if dirty:
            raise SystemExit(f"Keep or move local dependency edits before checking out: {path}")
        git("-C", path, "checkout", "--detach", entry["commit"])


if __name__ == "__main__":
    main()
