#!/usr/bin/env python3
from __future__ import annotations
import argparse, fnmatch, json, os, shutil, subprocess, sys, tempfile, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable
DEFAULT_TIMEOUT = 8.0
DEFAULT_BITS_TOL = 6  # ~10% of 64 bits for a 64-bit hash

EXCLUDE = [
    "main.py","cluster_visualizers.py","install_requirements.py",
    "test_*.py","*_test.py","*__init__*.py","entropic_worms.c",
]
PREFERRED = ["*visualizer*.py","pygameeq.py","pygamemusicvisualizer*.py","warpfield_visualizer.py"]

SITE_SHIM = r"""
# sitecustomize shim: capture frames whenever pygame flips/updates
import os, time
try:
    import pygame
    import numpy as _np  # ensure numpy backend present for surfarray
except Exception:
    pygame = None

OUT = os.environ.get("FPRINT_OUT")
MAXF = int(os.environ.get("CAPTURE_FRAMES","24"))
MIN_DT = float(os.environ.get("CAPTURE_MIN_DT","0.06"))

# Write a boot marker so the parent knows we loaded
if OUT:
    try:
        with open(OUT, "a") as f:
            f.write("BOOT\\n")
    except Exception:
        pass

if pygame:
    _orig_flip = pygame.display.flip
    _orig_update = pygame.display.update
    _last = 0.0
    _count = 0

    def _ahash_64(surf):
        arr = pygame.surfarray.array3d(surf)
        g = (arr[:,:,0].astype('float32')*0.299 + arr[:,:,1]*0.587 + arr[:,:,2]*0.114)
        h, w = g.shape
        sh, sw = h//8, w//8
        if sh == 0 or sw == 0:
            return 0
        g = g[:sh*8, :sw*8].reshape(8, sh, 8, sw).mean(axis=(1,3))
        m = g.mean()
        bits = (g > m).astype('uint8').flatten()
        out = 0
        for b in bits:
            out = (out << 1) | int(b)
        return int(out)

    def _capture():
        nonlocal _last, _count
        if OUT and _count < MAXF:
            now = time.time()
            if now - _last >= MIN_DT:
                surf = pygame.display.get_surface()
                if surf:
                    try:
                        h = _ahash_64(surf)
                        with open(OUT, "a") as f:
                            f.write(f"{h}\\n")
                        _count += 1
                        _last = now
                    except Exception:
                        pass

    def flip_patch():
        _capture()
        return _orig_flip()

    def update_patch(*args, **kwargs):
        _capture()
        return _orig_update(*args, **kwargs)

    pygame.display.flip = flip_patch
    pygame.display.update = update_patch
"""

BOOTSTRAP = r"""
# bootstrap.py — ensure our sitecustomize shim loads, then run target
import os, runpy, sys
TARGET = sys.argv[1]
runpy.run_path(TARGET, run_name="__main__")
"""

@dataclass
class Fingerprint:
    script: Path
    hash64: int
    frames: int

def discover(include: List[str], exclude: List[str]) -> List[Path]:
    inc = include or ["*.py"]
    exc = (exclude or []) + EXCLUDE
    items: List[Path] = []
    for p in REPO.iterdir():
        if p.is_file() and p.suffix == ".py":
            if any(fnmatch.fnmatch(p.name, x) for x in exc): continue
            items.append(p)
    def pri(path: Path) -> Tuple[int, str]:
        return (0 if any(fnmatch.fnmatch(path.name, s) for s in PREFERRED) else 1, path.name.lower())
    items.sort(key=pri)
    return items

def run_with_shim(script: Path, timeout: float):
    # Use an explicit bootstrap so sitecustomize is definitely imported
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "sitecustomize.py").write_text(SITE_SHIM)
        (td_path / "bootstrap.py").write_text(BOOTSTRAP)
        out_file = td_path / "hashes.txt"
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": f"{td}:{env.get('PYTHONPATH','')}",
            "FPRINT_OUT": str(out_file),
            "CAPTURE_FRAMES": "24",
            "CAPTURE_MIN_DT": "0.06",
        })
        try:
            proc = subprocess.Popen(
                [PYTHON, str(td_path / "bootstrap.py"), str(script)],
                cwd=str(REPO), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            return [], f"spawn_error: {e}"
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try: proc.terminate(); time.sleep(0.2)
            except Exception: pass
            try: proc.kill()
            except Exception: pass

        if out_file.exists():
            lines = [ln.strip() for ln in out_file.read_text().splitlines() if ln.strip()]
            # if only BOOT markers present, treat as no frames but shim loaded
            hashes = [int(x) for x in lines if x != "BOOT"] if lines else []
            return hashes, ("ok" if lines else "boot_no_frames")
        return [], "no_output"

def majority_hash(hashes: List[int]) -> int:
    if not hashes: return 0
    out = 0
    for bit in range(63, -1, -1):
        ones = sum((h >> bit) & 1 for h in hashes)
        zeros = len(hashes) - ones
        out = (out << 1) | (1 if ones >= zeros else 0)
    return out

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def cluster(fps: List[Fingerprint], bits_tol: int) -> List[List[Fingerprint]]:
    groups: List[List[Fingerprint]] = []
    for fp in fps:
        placed = False
        for g in groups:
            if hamming(fp.hash64, g[0].hash64) <= bits_tol:
                g.append(fp); placed = True; break
        if not placed:
            groups.append([fp])
    return groups

def choose_representative(group: List[Fingerprint]) -> Fingerprint:
    for pat in PREFERRED:
        for fp in group:
            if fnmatch.fnmatch(fp.script.name, pat): return fp
    return sorted(group, key=lambda x: (len(x.script.name), x.script.name))[0]

def write_reports(fps: List[Fingerprint], groups: List[List[Fingerprint]]):
    data = {
        "scripts": [{"script": fp.script.name, "hash64": hex(fp.hash64), "frames": fp.frames} for fp in fps],
        "clusters": [{"representative": choose_representative(g).script.name,
                      "members": [fp.script.name for fp in g]} for g in groups]
    }
    (REPO / "clusters.json").write_text(json.dumps(data, indent=2))
    with open(REPO / "clusters.csv", "w") as f:
        f.write("script,hash64,frames,cluster_rep\n")
        for g in groups:
            rep = choose_representative(g).script.name
            for fp in g:
                f.write(f"{fp.script.name},{hex(fp.hash64)},{fp.frames},{rep}\n")

def do_merge(groups: List[List[Fingerprint]]):
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = REPO / f"backups_cluster_{ts}"
    backup.mkdir(parents=True, exist_ok=True)
    consolidated = REPO / "consolidated"
    consolidated.mkdir(exist_ok=True)
    for g in groups:
        if len(g) == 1: continue
        rep = choose_representative(g)
        rep_dst = consolidated / rep.script.name
        if not rep_dst.exists(): shutil.copy2(rep.script, rep_dst)
        for fp in g:
            if fp.script == rep.script: continue
            shutil.copy2(fp.script, backup / fp.script.name)
            fp.script.write_text(f"""#!/usr/bin/env python3
# Auto-generated consolidation stub. Original backed up in {backup.name}
import runpy, os, sys
REP = os.path.join(os.path.dirname(__file__), 'consolidated', '{rep.script.name}')
if not os.path.exists(REP):
    sys.stderr.write("Representative not found: {rep.script.name}\\n"); sys.exit(2)
runpy.run_path(REP, run_name='__main__')
""")
            fp.script.chmod(0o755)
    print(f"Backups in: {backup}")
    print(f"Consolidated reps in: {consolidated}")

def main():
    ap = argparse.ArgumentParser(description="Fingerprint + cluster similar visualizer scripts")
    ap.add_argument("--include", action="append", default=[], help="Glob(s) to include (default: *.py)")
    ap.add_argument("--exclude", action="append", default=[], help="Glob(s) to exclude")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-script run timeout (s)")
    ap.add_argument("--bits", type=int, default=DEFAULT_BITS_TOL, help="Hamming tolerance in bits (10%≈6)")
    ap.add_argument("--merge", action="store_true", help="Replace near-duplicates with stubs to a representative")
    args = ap.parse_args()

    scripts = discover(args.include or ["*.py"], args.exclude)
    if not scripts:
        print("No scripts found."); return 1

    fps: List[Fingerprint] = []
    for s in scripts:
        print(f"[fingerprint] {s.name}…", end="", flush=True)
        hashes, status = run_with_shim(s, timeout=args.timeout)
        if not hashes:
            print(f" none ({status})"); continue
        rep = majority_hash(hashes)
        fps.append(Fingerprint(s, rep, len(hashes)))
        print(f" {len(hashes)} frames, hash={hex(rep)}")

    if not fps:
        print("No fingerprints captured."); return 1

    groups = cluster(fps, bits_tol=args.bits)
    write_reports(fps, groups)

    print("\nClusters:")
    for g in groups:
        names = ", ".join(fp.script.name for fp in g)
        print(f"  [{len(g)}] {names}")

    if args.merge:
        do_merge(groups)
        print("\nConsolidation complete.")

    print("\nWrote clusters.json and clusters.csv")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
