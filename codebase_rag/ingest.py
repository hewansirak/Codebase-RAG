"""Clone a GitHub repository and walk its source files."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".ipynb", ".java", ".c", ".h", ".cpp", ".hpp",
    ".go", ".rs", ".vue", ".swift", ".kt", ".rb", ".php", ".cs", ".md",
}  # fmt: skip
IGNORED_DIRS = {
    "node_modules", "venv", ".venv", "env", "dist", "build", ".git", "__pycache__",
    ".next", ".vscode", "vendor", "target", ".idea", "coverage",
}  # fmt: skip
MAX_FILE_BYTES = 300_000

_GITHUB_URL = re.compile(r"^(?:https?://)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


def parse_github_url(url: str) -> tuple[str, str]:
    """Return (owner, repo) for a github.com URL, or raise ValueError."""
    match = _GITHUB_URL.match(url.strip())
    if not match:
        raise ValueError(f"Not a GitHub repository URL: {url!r}")
    return match.group(1), match.group(2)


def clone_repository(url: str, repos_dir: Path) -> Path:
    """Shallow-clone `url` into repos_dir (or fast-forward an existing clone)."""
    owner, name = parse_github_url(url)
    dest = repos_dir / f"{owner}__{name}"
    if (dest / ".git").exists():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=True, capture_output=True)
        return dest
    repos_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", f"https://github.com/{owner}/{name}.git", str(dest)],
        check=True,
        capture_output=True,
    )
    return dest


def _notebook_source(raw: str) -> str:
    """Flatten a Jupyter notebook to its code cells."""
    try:
        cells = json.loads(raw).get("cells", [])
    except json.JSONDecodeError:
        return ""
    return "\n\n".join("".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code")


def iter_source_files(root: Path) -> Iterator[tuple[str, str]]:
    """Yield (posix relative path, text) for every supported, readable file under root."""
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in rel.parts[:-1]):
            continue
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if path.suffix.lower() == ".ipynb":
            text = _notebook_source(text)
        if text.strip():
            yield rel.as_posix(), text
