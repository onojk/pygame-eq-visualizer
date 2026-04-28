#!/usr/bin/env python3
"""
Mark.py — open each visualizer once; mark:
  m = merge, n = no-merge, s = skip, q = quit
Closes the window as soon as you choose. Writes marks_review.json/csv.

Default set = unique programs only:
  - consolidated_code/*.py
  - non-stub .py in repo root (excludes maint scripts)
Use --all to include everything. --include/--exclude to filter.
"""
from __future__ import annotations
import argparse, fnmatch, json, os, subprocess, sys, time
from pathlib import Path
from typing import List, Dict

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable
STUB_MARKER = "Auto-generated code-similarity stub"

DEFAULT_EXCLUDES = {
  "Mark.py","review_and_mark.py","main.py",
  "cluster_by_code.py","cluster_visualizers.py",
  "install_requirements.py","test_moviepy.py","test_opengl.py",
}

def is_stub(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = "".join([next(f) for _ in range(3)])
        return STUB_MARKER in head
    except Exception:
        return False

def discover(include: List[str], exclude: List[str], uniques_only: bool) -> List[Path]:
    items: List[Path] = []
    if uniques_only:
        cdir = REPO / "consolidated_code"
        if cdir.exists():
            items += sorted(cdir.glob("*.py"))
        for p in sorted(REPO.glob("*.py")):
            if p.name in DEFAULT_EXCLUDES: continue
            if p.suffix != ".py": continue
            if is_stub(p): continue
            items.append(p)
    else:
        items += sorted(REPO.glob("*.py"))
        cdir = REPO / "consolidated_code"
        if cdir.exists():
            items += sorted(cdir.glob("*.py"))

    # apply include/exclude
    if include:
        items = [p for p in items if any(fnmatch.fnmatch(p.name, g) or fnmatch.fnmatch(str(p), g) for g in include)]
    if exclude:
        items = [p for p in items if not any(fnmatch.fnmatch(p.name, g) or fnmatch.fnmatch(str(p), g) for g in exclude)]

    # unique, sorted
    seen, out = set(), []
    for p in items:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k); out.append(p)
    return sorted(out, key=lambda x: (x.parent.name, x.name.lower()))

def launch(path: Path) -> subprocess.Popen:
    return subprocess.Popen([PYTHON, str(path)], cwd=str(path.parent),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def terminate(proc: subprocess.Popen, grace: float = 0.5) -> None:
    if proc.poll() is not None: return
    try: proc.terminate()
    except Exception: pass
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < grace:
        time.sleep(0.05)
    if proc.poll() is None:
        try: proc.kill()
        except Exception: pass

def prompt_choice() -> str:
    while True:
        try:
            ch = input("  Mark this as [m]erge / [n]o-merge / [s]kip / [q]uit: ").strip().lower()
        except EOFError:
            ch = "s"
        if ch in ("m","n","s","q"):
            return ch
        print("  Please type m, n, s, or q, then Enter.")

def main():
    ap = argparse.ArgumentParser(description="Open scripts one-by-one and mark M/N/S/Q")
    ap.add_argument("--include", action="append", default=[], help="Glob(s) to include")
    ap.add_argument("--exclude", action="append", default=[], help="Glob(s) to exclude")
    ap.add_argument("--all", action="store_true", help="Review all .py (not just uniques)")
    args = ap.parse_args()

    items = discover(args.include, args.exclude, uniques_only=not args.all)
    if not items:
        print("No items to review."); return 1

    print(f"\nReviewing {len(items)} items. Controls: [M]erge  [N]o-merge  [S]kip  [Q]uit")
    marks: Dict[str,str] = {}

    for i, path in enumerate(items, 1):
        rel = path.relative_to(REPO)
        print(f"\n[{i}/{len(items)}] Launching: {rel}")
        proc = launch(path)
        ch = prompt_choice()  # <-- blocks until you choose
        terminate(proc)

        if ch == "q":
            print("Quitting early…")
            break
        if ch in ("m","n"):
            marks[str(rel)] = ch  # record only m/n; skips aren’t recorded

    # summary
    m_list = [k for k,v in marks.items() if v=="m"]
    n_list = [k for k,v in marks.items() if v=="n"]
    print("\nSummary:")
    print("  Merge (m):")
    for x in sorted(m_list): print("   -", x)
    print("  No-merge (n):")
    for x in sorted(n_list): print("   -", x)

    # save
    out_json = REPO / "marks_review.json"
    out_csv  = REPO / "marks_review.csv"
    with open(out_json, "w") as f: json.dump(marks, f, indent=2)
    with open(out_csv, "w") as f:
        f.write("file,mark\n")
        for k,v in marks.items(): f.write(f"{k},{v}\n")
    print(f"\nSaved: {out_json}\nSaved: {out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
