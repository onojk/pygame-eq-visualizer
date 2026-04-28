#!/usr/bin/env python3
"""
cluster_by_code.py — compare Pygame visualizer scripts by SOURCE CODE similarity
and consolidate near-duplicates. No runtime window capture; this works purely
on the text of each script.

Method
- Read each candidate .py file, strip comments/strings, normalize identifiers.
- Token-shingle (k=5 by default) and compute a 128‑bit SimHash per file.
- Cluster files whose SimHash Hamming distance <= threshold bits.
  For a 128-bit hash, 10% ≈ 12–13 bits (default 13).
- Optionally --merge: back up originals, pick a representative per cluster,
  and replace others with small stubs that delegate to the representative.

Usage
  python3 cluster_by_code.py                                # report only
  python3 cluster_by_code.py --include "*visualizer*.py"     # limit scope
  python3 cluster_by_code.py --bits 13 --merge               # 10% tol & merge

Outputs
- code_clusters.json / code_clusters.csv with per-file hash and cluster rep
- backups_code_<timestamp>/ and consolidated_code/ if --merge is used

Notes
- This is code-level similarity, so visually different programs that share a
  lot of scaffolding may be grouped unless you tighten --bits or increase
  shingle size (-k).
- Representative is chosen by preferred name pattern if available, otherwise
  shortest filename, then lexicographically.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import os
import re
import shutil
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

REPO = Path(__file__).resolve().parent
PREFERRED = [
    "*visualizer*.py",
    "pygameeq.py",
    "pygamemusicvisualizer*.py",
    "warpfield_visualizer.py",
]
EXCLUDE = [
    "main.py",
    "cluster_visualizers.py",
    "cluster_by_code.py",
    "install_requirements.py",
    "*__init__*.py",
    "test_*.py",
    "*_test.py",
]

@dataclass
class FileSig:
    path: Path
    simhash: int
    tokens: int

# ---------------- Normalization ----------------

_STRING_RE = re.compile(r"(?s)('''.*?'''|\"\"\".*?\"\"\"|'[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\")")
_COMMENT_RE = re.compile(r"(?m)#.*$")
_WS_RE = re.compile(r"\s+")

_PUNCT_TRANS = str.maketrans({c: ' ' for c in string.punctuation})

def strip_comments_and_strings(src: str) -> str:
    src = _STRING_RE.sub('""', src)        # keep placeholders so layout remains similar
    src = _COMMENT_RE.sub('', src)
    return src

def normalize_identifiers(src: str) -> str:
    # very light normalization: replace variable-like tokens with generic placeholders
    # keep keywords and numbers
    try:
        import tokenize
        from io import StringIO
        out = []
        g = tokenize.generate_tokens(StringIO(src).readline)
        for tok_type, tok_str, *_ in g:
            if tok_type == tokenize.NAME:
                if tok_str in {
                    'def','class','import','from','as','return','if','elif','else','for','while','try','except','finally','with','lambda','yield','pass','break','continue','in','is','and','or','not','global','nonlocal','assert','del','raise','True','False','None'
                }:
                    out.append(tok_str)
                else:
                    out.append('ID')
            elif tok_type == tokenize.NUMBER:
                out.append('NUM')
            elif tok_type == tokenize.STRING:
                out.append('STR')
            elif tok_type == tokenize.NEWLINE or tok_type == tokenize.NL:
                out.append('\n')
            else:
                out.append(tok_str)
        return ''.join(out)
    except Exception:
        return src

def tokenize_for_shingles(src: str) -> List[str]:
    # split on punctuation/whitespace, collapse empties
    txt = src.translate(_PUNCT_TRANS)
    txt = _WS_RE.sub(' ', txt)
    toks = [t for t in txt.strip().split(' ') if t]
    return toks

# ---------------- SimHash ----------------

def simhash_128(tokens: Iterable[str]) -> int:
    # 128-bit simhash: hash each token to 128-bit and add/subtract bit weights
    v = [0] * 128
    for tok in tokens:
        h = int.from_bytes(hashlib.blake2b(tok.encode('utf-8'), digest_size=16).digest(), 'big')
        for i in range(128):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(127, -1, -1):
        out = (out << 1) | (1 if v[i] >= 0 else 0)
    return out

# ---------------- Pipeline ----------------

def shingle(tokens: List[str], k: int) -> List[str]:
    if k <= 1:
        return tokens
    return [' '.join(tokens[i:i+k]) for i in range(max(0, len(tokens)-k+1))]

def build_signature(path: Path, k: int) -> FileSig:
    src = path.read_text(errors='ignore')
    src = strip_comments_and_strings(src)
    src = normalize_identifiers(src)
    toks = tokenize_for_shingles(src)
    toks = shingle(toks, k)
    return FileSig(path, simhash_128(toks), len(toks))

# ---------------- Clustering ----------------

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def discover(include: List[str], exclude: List[str]) -> List[Path]:
    inc = include or ['*.py']
    exc = (exclude or []) + EXCLUDE
    files: List[Path] = []
    for p in REPO.iterdir():
        if p.is_file() and p.suffix == '.py':
            if any(fnmatch.fnmatch(p.name, e) for e in exc):
                continue
            if not any(fnmatch.fnmatch(p.name, i) for i in inc):
                # default include *.py anyway
                pass
            files.append(p)
    files.sort(key=lambda x: x.name.lower())
    return files

def cluster_files(sigs: List[FileSig], bits: int) -> List[List[FileSig]]:
    groups: List[List[FileSig]] = []
    for fs in sigs:
        placed = False
        for g in groups:
            if hamming(fs.simhash, g[0].simhash) <= bits:
                g.append(fs); placed = True; break
        if not placed:
            groups.append([fs])
    return groups

def choose_rep(group: List[FileSig]) -> FileSig:
    for pat in PREFERRED:
        for fs in group:
            if fnmatch.fnmatch(fs.path.name, pat):
                return fs
    return sorted(group, key=lambda fs: (len(fs.path.name), fs.path.name))[0]

# ---------------- Merge ----------------

def merge_groups(groups: List[List[FileSig]]):
    ts = time.strftime('%Y%m%d_%H%M%S')
    backup = REPO / f'backups_code_{ts}'
    backup.mkdir(parents=True, exist_ok=True)
    outdir = REPO / 'consolidated_code'
    outdir.mkdir(exist_ok=True)

    for g in groups:
        if len(g) == 1:
            continue
        rep = choose_rep(g)
        rep_dst = outdir / rep.path.name
        if not rep_dst.exists():
            shutil.copy2(rep.path, rep_dst)
        for fs in g:
            if fs.path == rep.path:
                continue
            shutil.copy2(fs.path, backup / fs.path.name)
            fs.path.write_text(f"""#!/usr/bin/env python3
# Auto-generated code-similarity stub. Original backed up in {backup.name}
import runpy, os, sys
REP = os.path.join(os.path.dirname(__file__), 'consolidated_code', '{rep.path.name}')
if not os.path.exists(REP):
    sys.stderr.write('Representative not found: {rep.path.name}\n'); sys.exit(2)
runpy.run_path(REP, run_name='__main__')
""")
            fs.path.chmod(0o755)
    print(f"Backups in: {backup}")
    print(f"Representatives in: {outdir}")

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description='Cluster and merge near-duplicate Python scripts by code similarity')
    ap.add_argument('--include', action='append', default=[], help='Glob(s) to include (default: *.py)')
    ap.add_argument('--exclude', action='append', default=[], help='Glob(s) to exclude')
    ap.add_argument('--bits', type=int, default=13, help='Hamming distance threshold in bits (128-bit hash). 10%≈12-13')
    ap.add_argument('-k', '--shingle', type=int, default=5, help='Shingle size (k tokens per shingle)')
    ap.add_argument('--merge', action='store_true', help='Replace near-duplicates with stubs to a representative')
    args = ap.parse_args()

    files = discover(args.include or ['*.py'], args.exclude)
    if not files:
        print('No files found.'); return 1

    sigs = [build_signature(p, args.shingle) for p in files]

    # write report files
    import csv, json
    data = {
        'files': [{'file': s.path.name, 'hash': hex(s.simhash), 'tokens': s.tokens} for s in sigs]
    }
    (REPO / 'code_clusters.json').write_text(json.dumps(data, indent=2))

    groups = cluster_files(sigs, args.bits)
    with open(REPO / 'code_clusters.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file','hash','tokens','cluster_rep'])
        for g in groups:
            rep = choose_rep(g)
            for s in g:
                w.writerow([s.path.name, hex(s.simhash), s.tokens, rep.path.name])

    print('\nClusters:')
    for g in groups:
        rep = choose_rep(g).path.name
        names = ', '.join(s.path.name for s in g)
        print(f'  [{len(g)}] rep={rep} :: {names}')

    if args.merge:
        merge_groups(groups)
        print('\nConsolidation complete.')

    print('\nWrote code_clusters.json and code_clusters.csv')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
