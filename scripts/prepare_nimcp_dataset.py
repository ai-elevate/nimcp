#!/usr/bin/env python3
"""
Prepare NIMCP Codebase as Training Dataset
==========================================

WHAT: Package the NIMCP source code and documentation into HuggingFace-compatible
      datasets for self-training Athena on its own codebase.
WHY:  Self-knowledge — Athena should understand the system it runs on.
HOW:  Walk the source tree, chunk files into training examples, save as
      Arrow datasets loadable via `load_dataset("path")`.

Produces 3 datasets:
  1. nimcp_source  — C/H source code (functions, structs, comments)
  2. nimcp_docs    — Markdown documentation
  3. nimcp_scripts — Python training/tooling scripts
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "nimcp_local"


def chunk_code_file(filepath: Path, max_chunk_lines: int = 80) -> list:
    """Split a C/H file into function-level or chunk-level examples."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = text.split("\n")
    if len(lines) < 3:
        return []

    rel_path = str(filepath.relative_to(PROJECT_ROOT))
    chunks = []

    # Try to split on function boundaries (lines starting with no indent
    # that contain '(' and aren't preprocessor directives)
    func_starts = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Function definition heuristic: non-indented, has '(', not #define/include
        if (line and not line[0].isspace() and "(" in line
                and not stripped.startswith("#")
                and not stripped.startswith("//")
                and not stripped.startswith("/*")
                and not stripped.startswith("*")):
            func_starts.append(i)

    if len(func_starts) >= 2:
        # Split at function boundaries
        for idx, start in enumerate(func_starts):
            end = func_starts[idx + 1] if idx + 1 < len(func_starts) else len(lines)
            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines).strip()
            if len(chunk_text) > 50:
                chunks.append({
                    "text": f"// File: {rel_path} (lines {start+1}-{end})\n{chunk_text}",
                    "label": f"nimcp_source:{rel_path}",
                })
    else:
        # Small file or header — take as one chunk or split by line count
        for i in range(0, len(lines), max_chunk_lines):
            chunk_lines = lines[i:i + max_chunk_lines]
            chunk_text = "\n".join(chunk_lines).strip()
            if len(chunk_text) > 50:
                chunks.append({
                    "text": f"// File: {rel_path} (lines {i+1}-{i+len(chunk_lines)})\n{chunk_text}",
                    "label": f"nimcp_source:{rel_path}",
                })

    return chunks


def chunk_markdown_file(filepath: Path) -> list:
    """Split a markdown file by ## headings into training examples."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    if len(text.strip()) < 50:
        return []

    rel_path = str(filepath.relative_to(PROJECT_ROOT))
    chunks = []

    # Split on ## headings
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if len(section) < 50:
            continue
        # Cap sections at ~4000 chars
        if len(section) > 4000:
            # Sub-split on ### or paragraph breaks
            sub_parts = re.split(r"(?=^### |\n\n)", section, flags=re.MULTILINE)
            for part in sub_parts:
                part = part.strip()
                if len(part) > 50:
                    chunks.append({
                        "text": f"<!-- File: {rel_path} -->\n{part[:4000]}",
                        "label": f"nimcp_docs:{rel_path}",
                    })
        else:
            chunks.append({
                "text": f"<!-- File: {rel_path} -->\n{section}",
                "label": f"nimcp_docs:{rel_path}",
            })

    return chunks


def chunk_python_file(filepath: Path, max_chunk_lines: int = 60) -> list:
    """Split a Python file by class/function definitions."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = text.split("\n")
    if len(lines) < 3:
        return []

    rel_path = str(filepath.relative_to(PROJECT_ROOT))
    chunks = []

    # Split on top-level class/def
    def_starts = []
    for i, line in enumerate(lines):
        if re.match(r"^(class |def )", line):
            def_starts.append(i)

    if len(def_starts) >= 2:
        # Include file header (imports, docstring) as first chunk
        if def_starts[0] > 3:
            header = "\n".join(lines[:def_starts[0]]).strip()
            if len(header) > 50:
                chunks.append({
                    "text": f"# File: {rel_path} (header)\n{header}",
                    "label": f"nimcp_scripts:{rel_path}",
                })

        for idx, start in enumerate(def_starts):
            end = def_starts[idx + 1] if idx + 1 < len(def_starts) else len(lines)
            chunk_text = "\n".join(lines[start:end]).strip()
            if len(chunk_text) > 50:
                chunks.append({
                    "text": f"# File: {rel_path} (lines {start+1}-{end})\n{chunk_text}",
                    "label": f"nimcp_scripts:{rel_path}",
                })
    else:
        # Small file — one chunk
        for i in range(0, len(lines), max_chunk_lines):
            chunk_lines = lines[i:i + max_chunk_lines]
            chunk_text = "\n".join(chunk_lines).strip()
            if len(chunk_text) > 50:
                chunks.append({
                    "text": f"# File: {rel_path} (lines {i+1}-{i+len(chunk_lines)})\n{chunk_text}",
                    "label": f"nimcp_scripts:{rel_path}",
                })

    return chunks


def collect_files(root: Path, extensions: set) -> list:
    """Recursively collect files with given extensions, skipping build/."""
    files = []
    skip_dirs = {"build", ".git", "node_modules", "__pycache__", ".venv", "checkpoints"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in sorted(filenames):
            if any(fname.endswith(ext) for ext in extensions):
                files.append(Path(dirpath) / fname)
    return files


def save_jsonl(examples: list, output_path: Path):
    """Save examples as JSONL (loadable via load_dataset('json', data_files=...))."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved {len(examples):,} examples → {output_path}")


def main():
    print("Preparing NIMCP codebase datasets...")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Output dir:   {OUTPUT_DIR}")

    # 1. C/H source code
    print("\n[1/3] Collecting C/H source files...")
    c_files = collect_files(PROJECT_ROOT / "src", {".c", ".h"})
    h_files = collect_files(PROJECT_ROOT / "include", {".h"})
    all_source = c_files + h_files
    print(f"  Found {len(all_source)} source files")

    source_examples = []
    for f in all_source:
        source_examples.extend(chunk_code_file(f))
    save_jsonl(source_examples, OUTPUT_DIR / "nimcp_source" / "train.jsonl")

    # 2. Markdown documentation
    print("\n[2/3] Collecting documentation...")
    doc_files = collect_files(PROJECT_ROOT / "docs", {".md"})
    # Also include top-level docs
    for md in PROJECT_ROOT.glob("*.md"):
        doc_files.append(md)
    print(f"  Found {len(doc_files)} doc files")

    doc_examples = []
    for f in doc_files:
        doc_examples.extend(chunk_markdown_file(f))
    save_jsonl(doc_examples, OUTPUT_DIR / "nimcp_docs" / "train.jsonl")

    # 3. Python scripts
    print("\n[3/3] Collecting Python scripts...")
    py_files = collect_files(PROJECT_ROOT / "scripts", {".py"})
    # Also include example Python files
    py_files.extend(collect_files(PROJECT_ROOT / "examples", {".py"}))
    # Include test Python files
    py_files.extend(collect_files(PROJECT_ROOT / "test", {".py"}))
    print(f"  Found {len(py_files)} Python files")

    script_examples = []
    for f in py_files:
        script_examples.extend(chunk_python_file(f))
    save_jsonl(script_examples, OUTPUT_DIR / "nimcp_scripts" / "train.jsonl")

    # Summary
    total = len(source_examples) + len(doc_examples) + len(script_examples)
    print(f"\n{'='*60}")
    print(f"NIMCP Dataset Preparation Complete")
    print(f"  Source code:    {len(source_examples):>6,} examples")
    print(f"  Documentation:  {len(doc_examples):>6,} examples")
    print(f"  Python scripts: {len(script_examples):>6,} examples")
    print(f"  Total:          {total:>6,} examples")
    print(f"  Output:         {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
