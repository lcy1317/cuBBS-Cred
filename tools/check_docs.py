#!/usr/bin/env python3
"""Check local Markdown links, Cargo readmes and retired documentation paths."""

from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\]\(<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\)")


def git_paths(*args):
    output = subprocess.check_output(["git", *args, "-z"], cwd=ROOT)
    return [Path(name.decode()) for name in output.split(b"\0") if name]


def main():
    # Include new docs before they are staged; never traverse ignored archives.
    tracked = git_paths("ls-files", "--cached", "--others", "--exclude-standard")
    documents = {ROOT / p for p in tracked if p.suffix == ".md"}
    documents.update((ROOT / "docs").rglob("*.md"))
    errors = []
    checked = 0
    for path in sorted(documents):
        if not path.is_file():
            continue
        checked += 1
        fenced = False
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
                continue
            if fenced:
                continue
            for target in LINK.findall(line):
                url = urlsplit(target)
                if url.scheme or url.netloc or not url.path:
                    continue
                if not (path.parent / unquote(url.path)).exists():
                    errors.append(f"{path.relative_to(ROOT)}:{number}: missing {target}")

    for relative in tracked:
        path = ROOT / relative
        if relative.name != "Cargo.toml" or not path.exists():
            continue
        for readme in re.findall(r'^readme\s*=\s*"([^"]+)"', path.read_text(), re.M):
            if not (path.parent / readme).is_file():
                errors.append(f"{relative}: missing readme {readme}")

    retired = git_paths("diff", "--name-only", "--diff-filter=D")
    retired_docs = [p for p in retired if p.suffix in (".md", ".rule")]
    sources = {ROOT / p for p in tracked}
    sources.update((ROOT / "docs").rglob("*"))
    for path in sorted(sources):
        if not path.is_file() or path.suffix not in (".md", ".py", ".sh", ".toml", ".rs"):
            continue
        content = path.read_text()
        for old in retired_docs:
            if old.as_posix() in content:
                errors.append(f"{path.relative_to(ROOT)}: references retired {old}")

    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"PASS: {checked} Markdown files, local link targets, Cargo readmes and retired paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
