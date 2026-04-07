"""Split source files into retrievable chunks that carry file path and line numbers.

Python files are split with the `ast` module into functions, methods and classes.
Other languages are split at top-level declaration boundaries found with regular
expressions (not a full parser). Anything left over, and any oversized chunk, falls
back to overlapping line windows.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass

MAX_CHUNK_LINES = 80
WINDOW_OVERLAP = 10

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".ipynb": "python", ".java": "java", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".go": "go", ".rs": "rust", ".vue": "vue",
    ".swift": "swift", ".kt": "kotlin", ".rb": "ruby", ".php": "php", ".cs": "csharp",
    ".md": "markdown",
}  # fmt: skip

# A top-level declaration: starts in column 0 with a keyword that introduces a named symbol.
_DECL = re.compile(
    r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?"
    r"(?:(?P<kw>function\*?|class|interface|enum|struct|trait|impl|func|fn|def|type|"
    r"public|private|protected|static|final|abstract)\s+(?:[\w<>\[\],.*&\s]*?\s)?(?P<name>[A-Za-z_]\w*)"
    r"|(?:const|let|var)\s+(?P<cname>[A-Za-z_]\w*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>)"
)
_TYPE_DECL = re.compile(r"\s*(export\s+)?(default\s+)?(class|interface|struct|enum)\b")


@dataclass
class Chunk:
    repo: str
    path: str
    start_line: int
    end_line: int
    language: str
    kind: str  # function | method | class | module | block | doc
    symbol: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.repo}:{self.path}:{self.start_line}-{self.end_line}"

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def embedding_text(self) -> str:
        """Text that is embedded and indexed: location + symbol give lexical/semantic hints."""
        return f"{self.path} {self.symbol}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def detect_language(path: str) -> str:
    dot = path.rfind(".")
    return LANGUAGES.get(path[dot:].lower(), "text") if dot != -1 else "text"


def _make(repo, path, lang, kind, symbol, lines, start) -> Chunk:
    return Chunk(repo, path, start, start + len(lines) - 1, lang, kind, symbol, "\n".join(lines))


def _windows(repo, path, lang, kind, symbol, lines, start) -> list[Chunk]:
    """Split `lines` (which begin at 1-based line `start`) into overlapping windows."""
    if not any(line.strip() for line in lines):
        return []
    if len(lines) <= MAX_CHUNK_LINES:
        return [_make(repo, path, lang, kind, symbol, lines, start)]
    out, step = [], MAX_CHUNK_LINES - WINDOW_OVERLAP
    for offset in range(0, len(lines), step):
        part = lines[offset : offset + MAX_CHUNK_LINES]
        if any(line.strip() for line in part):
            out.append(_make(repo, path, lang, kind, symbol, part, start + offset))
        if offset + MAX_CHUNK_LINES >= len(lines):
            break
    return out


def _python_symbols(tree: ast.Module) -> list[tuple[str, str, int, int]]:
    """(kind, qualified name, first line, last line) for functions, methods and class headers."""
    found: list[tuple[str, str, int, int]] = []
    funcs = (ast.FunctionDef, ast.AsyncFunctionDef)

    def first_line(node) -> int:
        decorators = [d.lineno for d in getattr(node, "decorator_list", [])]
        return min([node.lineno, *decorators])

    for node in tree.body:
        if isinstance(node, funcs):
            found.append(("function", node.name, first_line(node), node.end_lineno))
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, funcs)]
            header_end = (first_line(methods[0]) - 1) if methods else node.end_lineno
            found.append(("class", node.name, first_line(node), header_end))
            for m in methods:
                found.append(("method", f"{node.name}.{m.name}", first_line(m), m.end_lineno))
    return found


def _chunk_python(repo: str, path: str, lines: list[str]) -> list[Chunk] | None:
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return None
    chunks: list[Chunk] = []
    covered = [False] * (len(lines) + 2)
    for kind, name, first, last in _python_symbols(tree):
        for n in range(first, last + 1):
            covered[n] = True
        chunks += _windows(repo, path, "python", kind, name, lines[first - 1 : last], first)
    # Module-level code (imports, constants, scripts) between symbols.
    n = 1
    while n <= len(lines):
        if covered[n]:
            n += 1
            continue
        start = n
        while n <= len(lines) and not covered[n]:
            n += 1
        chunks += _windows(repo, path, "python", "module", "<module>", lines[start - 1 : n - 1], start)
    return sorted(chunks, key=lambda c: c.start_line)


def _chunk_by_declarations(repo: str, path: str, lang: str, lines: list[str]) -> list[Chunk]:
    starts: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        match = _DECL.match(line)
        if match:
            starts.append((idx, match.group("name") or match.group("cname")))
    if not starts:
        return _windows(repo, path, lang, "block", "<file>", lines, 1)
    chunks: list[Chunk] = []
    if starts[0][0] > 0:
        chunks += _windows(repo, path, lang, "module", "<module>", lines[: starts[0][0]], 1)
    for i, (idx, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(lines)
        kind = "class" if _TYPE_DECL.match(lines[idx]) else "function"
        chunks += _windows(repo, path, lang, kind, name, lines[idx:end], idx + 1)
    return chunks


def _chunk_markdown(repo: str, path: str, lines: list[str]) -> list[Chunk]:
    starts = [i for i, line in enumerate(lines) if line.startswith("#")] or [0]
    if starts[0] != 0:
        starts.insert(0, 0)
    chunks: list[Chunk] = []
    for i, idx in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(lines)
        title = lines[idx].lstrip("# ").strip() or "<intro>"
        chunks += _windows(repo, path, "markdown", "doc", title, lines[idx:end], idx + 1)
    return chunks


def chunk_file(repo: str, path: str, text: str) -> list[Chunk]:
    """Chunk a single file. `path` is the repo-relative posix path."""
    lang = detect_language(path)
    lines = text.splitlines()
    if lang == "markdown":
        return _chunk_markdown(repo, path, lines)
    if lang == "python" and not path.endswith(".ipynb"):
        chunks = _chunk_python(repo, path, lines)
        if chunks is not None:
            return chunks
    if lang == "python":  # notebooks (flattened code cells) and unparsable files
        return _windows(repo, path, lang, "block", "<file>", lines, 1)
    return _chunk_by_declarations(repo, path, lang, lines)
