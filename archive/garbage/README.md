# archive/garbage/

Files that should not have been in the repo. Kept for transparency only — safe to delete.

## What's here

### PostScript dumps (~104 MB total)
- `pygame`, `pyaudio`, `np`, `math`, `sys`, `random`

These look like Python module names but they are actually **ImageMagick PostScript files**
(13–40 MB each). They were created — almost certainly by accident — when ImageMagick was
told to write to a file with no extension. They have nothing to do with Python modules.

```
$ file pygame
pygame: PostScript document text conforming DSC level 3.0, Level 1
$ head -c 50 pygame
%!PS-Adobe-3.0
%%Creator: (ImageMagick)
```

### Pip-typo files
- `=0.0.0`, `=0.23.0`, `=1.18.1`, `=1.3.0`, `=2.5.0`, `=4.0.0`, `=4.3.0`, `=4.5`, `=7.0.0`

Created by typos like `pip install foo >=1.18.1` — bash interprets `>=1.18.1` as redirection
to a file literally named `=1.18.1`. All empty except `=4.5` which has captured pip output.

### Editor crash files
- `nano.23866.save` — nano editor crash recovery file. Has malformed Python.

### Stale backups
- `main.py.bak` — three-line diff from current `main.py`. Outdated.
