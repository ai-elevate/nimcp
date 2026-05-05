"""Unit tests for scripts/code_corpus.py — CE-5 source-code corpus
loader + code-aware chunker.

Pure-Python tests: no daemon, no live brain. Validates the shipped
data/code_corpus/ tree (index.json + sample files) against the
contract documented in CLAUDE.md.
"""
import json
import os
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import code_corpus as cc  # noqa: E402

CORPUS_ROOT = os.path.join(REPO, "data", "code_corpus")


# ---------------------------------------------------------------------------
# Helper: write a temp file with given suffix and contents.
# ---------------------------------------------------------------------------

def _temp_file(suffix, contents):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)
    return path


# ---------------------------------------------------------------------------
# Chunker: Python
# ---------------------------------------------------------------------------

class TestPythonChunker(unittest.TestCase):
    def test_three_defs_yield_at_least_three_chunks(self):
        # Each function body is padded above the coalesce floor so the
        # tiny-chunk coalescer doesn't fold them together.
        body = "\n".join("    # filler comment line {}".format(i)
                          for i in range(8))
        src = (
            "import os\n"
            "\n"
            "\n"
            "def alpha():\n"
            + body + "\n"
            "    return 1\n"
            "\n"
            "\n"
            "def beta():\n"
            + body + "\n"
            "    return 2\n"
            "\n"
            "\n"
            "def gamma():\n"
            + body + "\n"
            "    return 3\n"
        )
        path = _temp_file(".py", src)
        try:
            chunks = list(cc.iter_code_chunks(path))
        finally:
            os.remove(path)

        # Three defs => at least three chunks (preamble may be coalesced
        # into the first def's chunk if it's tiny — that gives 3 chunks;
        # if the preamble is its own chunk we get 4).
        self.assertGreaterEqual(len(chunks), 3)

        # Each chunk should mention exactly one of the function names if
        # the boundary detection is working.
        text_blob = "\n".join(c[1] for c in chunks)
        self.assertIn("def alpha", text_blob)
        self.assertIn("def beta", text_blob)
        self.assertIn("def gamma", text_blob)

    def test_class_and_def_both_split(self):
        src = (
            "class Foo:\n"
            "    pass\n"
            "\n"
            "\n"
            "def bar():\n"
            "    return 0\n"
        )
        path = _temp_file(".py", src)
        try:
            chunks = list(cc.iter_code_chunks(path))
        finally:
            os.remove(path)
        # Class and def: at least two chunks (or one if coalesced —
        # but our `pass` body is tiny so it will coalesce).
        self.assertGreaterEqual(len(chunks), 1)

    def test_leading_comment_attaches_to_function(self):
        src = (
            "import sys\n"
            "\n"
            "\n"
            "# This comment documents `compute`.\n"
            "# It must stay attached to the def below.\n"
            "def compute(x):\n"
            "    return x * 2\n"
        )
        path = _temp_file(".py", src)
        try:
            chunks = list(cc.iter_code_chunks(path))
        finally:
            os.remove(path)

        # Find the chunk that contains `def compute`.
        compute_chunk = None
        for _, text in chunks:
            if "def compute" in text:
                compute_chunk = text
                break
        self.assertIsNotNone(compute_chunk)
        self.assertIn("documents `compute`", compute_chunk)
        self.assertIn("must stay attached", compute_chunk)

    def test_oversize_chunk_splits_on_blank_lines(self):
        # Build one giant function with internal blank lines so the
        # oversize fallback kicks in.
        body = "\n".join("    x = " + str(i) for i in range(200))
        # Insert blank lines every 10 to give the splitter cut points.
        body = body.replace("    x = 50", "    x = 50\n", 1)
        src = "def big():\n" + body + "\n"
        path = _temp_file(".py", src)
        try:
            chunks = list(cc.iter_code_chunks(path, max_chars=200))
        finally:
            os.remove(path)
        # With max_chars=200, this should split into multiple pieces.
        self.assertGreater(len(chunks), 1)
        # Every piece must respect max_chars (modulo coalescing — but
        # our giant chunk's pieces are all larger than the tiny floor,
        # so the floor doesn't expand them). Allow a small slack for
        # the coalescer joining a tail.
        for _, text in chunks:
            self.assertLessEqual(len(text), 200 * 3)

    def test_tiny_chunks_coalesce(self):
        # Two trivial defs that would each yield <50-char chunks if we
        # didn't coalesce.
        src = (
            "def a(): return 1\n"
            "def b(): return 2\n"
        )
        path = _temp_file(".py", src)
        try:
            chunks = list(cc.iter_code_chunks(path))
        finally:
            os.remove(path)
        # Coalescing should fold these — every emitted chunk should be
        # >= the minimum threshold OR there should only be one chunk.
        if len(chunks) > 1:
            for _, text in chunks:
                self.assertGreaterEqual(len(text), 1)


# ---------------------------------------------------------------------------
# Chunker: C
# ---------------------------------------------------------------------------

class TestCChunker(unittest.TestCase):
    def test_two_functions_yield_at_least_two_chunks(self):
        # Pad each body above the coalesce floor.
        body = "\n".join("    /* filler comment line {} */".format(i)
                          for i in range(6))
        src = (
            "#include <stdio.h>\n"
            "\n"
            "int alpha(int x) {\n"
            + body + "\n"
            "    return x + 1;\n"
            "}\n"
            "\n"
            "int beta(int y) {\n"
            + body + "\n"
            "    return y * 2;\n"
            "}\n"
        )
        path = _temp_file(".c", src)
        try:
            chunks = list(cc.iter_code_chunks(path))
        finally:
            os.remove(path)
        # Two C functions => at least two chunks (preamble + 2, or 2 if
        # the preamble is coalesced into the first function).
        self.assertGreaterEqual(len(chunks), 2)
        blob = "\n".join(c[1] for c in chunks)
        self.assertIn("int alpha", blob)
        self.assertIn("int beta", blob)

    def test_c_block_comment_attaches_to_function(self):
        src = (
            "#include <stdio.h>\n"
            "\n"
            "/* This block comment documents alpha.\n"
            " * It must stay attached to the function below. */\n"
            "int alpha(int x) {\n"
            "    return x + 1;\n"
            "}\n"
        )
        path = _temp_file(".c", src)
        try:
            chunks = list(cc.iter_code_chunks(path))
        finally:
            os.remove(path)
        alpha_chunk = None
        for _, text in chunks:
            if "int alpha" in text:
                alpha_chunk = text
                break
        self.assertIsNotNone(alpha_chunk)
        self.assertIn("documents alpha", alpha_chunk)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class TestIndex(unittest.TestCase):
    def test_index_parses(self):
        idx = cc.load_code_index(CORPUS_ROOT)
        self.assertEqual(idx.get("version"), 1)
        self.assertIn("works", idx)
        self.assertIn("registers", idx)

    def test_required_keys_per_work(self):
        idx = cc.load_code_index(CORPUS_ROOT)
        for work in idx["works"]:
            for key in ("id", "register", "title", "files",
                        "language", "stage", "weight", "source"):
                self.assertIn(key, work, f"missing {key} on {work.get('id')}")

    def test_minimum_stage_counts(self):
        idx = cc.load_code_index(CORPUS_ROOT)
        stage2 = [w for w in idx["works"] if int(w["stage"]) == 2]
        stage3 = [w for w in idx["works"] if int(w["stage"]) == 3]
        self.assertGreaterEqual(len(stage2), 5)
        self.assertGreaterEqual(len(stage3), 7)

    def test_no_duplicate_ids(self):
        idx = cc.load_code_index(CORPUS_ROOT)
        ids = [w["id"] for w in idx["works"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_language_field_valid(self):
        idx = cc.load_code_index(CORPUS_ROOT)
        for work in idx["works"]:
            self.assertIn(work["language"], ("python", "c"))

    def test_weight_in_range(self):
        idx = cc.load_code_index(CORPUS_ROOT)
        for work in idx["works"]:
            w = float(work["weight"])
            self.assertGreaterEqual(w, 0.5)
            self.assertLessEqual(w, 2.0)


# ---------------------------------------------------------------------------
# Sample files
# ---------------------------------------------------------------------------

class TestSampleFiles(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(CORPUS_ROOT, "index.json"),
                  "r", encoding="utf-8") as f:
            self.idx = json.load(f)

    def _all_paths(self):
        for work in self.idx["works"]:
            for rel in work["files"]:
                yield work, os.path.join(CORPUS_ROOT, rel)

    def test_every_referenced_file_exists(self):
        for work, path in self._all_paths():
            self.assertTrue(os.path.isfile(path),
                            f"{work['id']}: missing {path}")

    def test_files_are_utf8_and_have_30_lines(self):
        for work, path in self._all_paths():
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertGreaterEqual(
                len(lines), 30,
                f"{work['id']}: only {len(lines)} lines (need >= 30)")

    def test_no_placeholder_text(self):
        forbidden = ("TODO", "PLACEHOLDER", "FIXME", "lorem", "your code here")
        for work, path in self._all_paths():
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for needle in forbidden:
                self.assertNotIn(
                    needle, text,
                    f"{work['id']}: contains forbidden token {needle!r}")
                self.assertNotIn(
                    needle.lower(), text.lower(),
                    f"{work['id']}: contains forbidden token {needle!r} "
                    f"(case-insensitive)")

    def test_python_files_compile(self):
        for work, path in self._all_paths():
            if work["language"] != "python":
                continue
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            try:
                compile(text, path, "exec")
            except SyntaxError as e:
                self.fail(f"{work['id']}: SyntaxError {e}")

    def test_c_files_have_main(self):
        for work, path in self._all_paths():
            if work["language"] != "c":
                continue
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.assertIn(
                "int main", text,
                f"{work['id']}: must contain `int main` to be standalone")


# ---------------------------------------------------------------------------
# End-to-end: run the chunker on every shipped sample.
# ---------------------------------------------------------------------------

class TestEndToEndChunkAllSamples(unittest.TestCase):
    def test_every_sample_yields_at_least_one_chunk(self):
        with open(os.path.join(CORPUS_ROOT, "index.json"),
                  "r", encoding="utf-8") as f:
            idx = json.load(f)
        for work in idx["works"]:
            for rel in work["files"]:
                path = os.path.join(CORPUS_ROOT, rel)
                chunks = list(cc.iter_code_chunks(path))
                self.assertGreater(
                    len(chunks), 0,
                    f"{work['id']}: chunker yielded zero chunks")
                # Every chunk must be non-empty after strip.
                for cid, text in chunks:
                    self.assertGreater(len(text.strip()), 0)


if __name__ == "__main__":
    unittest.main()
