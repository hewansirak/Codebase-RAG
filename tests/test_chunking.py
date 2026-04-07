from codebase_rag.chunking import MAX_CHUNK_LINES, chunk_file

PY = '''import os

CONST = 1


def top(a):
    """Top-level function."""
    return a


class Greeter:
    """A greeter."""

    def hello(self):
        return "hi"

    @staticmethod
    def bye():
        return "bye"
'''


def test_python_functions_methods_and_classes_have_exact_lines():
    chunks = {c.symbol: c for c in chunk_file("o/r", "pkg/mod.py", PY)}
    assert chunks["top"].kind == "function"
    assert (chunks["top"].start_line, chunks["top"].end_line) == (6, 8)
    assert chunks["Greeter"].kind == "class"
    assert chunks["Greeter.hello"].kind == "method"
    # decorator line is included in the method chunk
    assert chunks["Greeter.bye"].text.startswith("    @staticmethod")
    assert chunks["<module>"].text.startswith("import os")


def test_chunks_cover_every_nonblank_line_once():
    chunks = chunk_file("o/r", "mod.py", PY)
    covered = [n for c in chunks for n in range(c.start_line, c.end_line + 1)]
    nonblank = [i for i, line in enumerate(PY.splitlines(), 1) if line.strip()]
    assert set(nonblank) <= set(covered)
    assert len(covered) == len(set(covered))


def test_syntax_error_falls_back_to_windows():
    chunks = chunk_file("o/r", "broken.py", "def (:\n  pass\n")
    assert len(chunks) == 1 and chunks[0].kind == "block"


def test_large_function_is_split_with_overlap():
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    chunks = chunk_file("o/r", "big.py", f"def big():\n{body}\n")
    assert len(chunks) > 1
    assert all(c.end_line - c.start_line + 1 <= MAX_CHUNK_LINES for c in chunks)
    assert chunks[1].start_line < chunks[0].end_line  # overlap


def test_typescript_declarations():
    ts = (
        "import x from 'y';\n\n"
        "export async function load(id: string) {\n  return 1;\n}\n\n"
        "export const save = async (a) => {\n  return a;\n};\n\n"
        "export class Repo {\n}\n"
    )
    chunks = {c.symbol: c for c in chunk_file("o/r", "src/a.ts", ts)}
    assert {"load", "save", "Repo"} <= set(chunks)
    assert chunks["Repo"].kind == "class"
    assert chunks["load"].start_line == 3


def test_markdown_split_by_headings():
    chunks = chunk_file("o/r", "README.md", "# A\ntext\n## B\nmore\n")
    assert [c.symbol for c in chunks] == ["A", "B"]
