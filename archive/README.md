# archive/

Material kept out of the live project but preserved for reference. Nothing in here is
imported by any working script. You can safely delete the entire `archive/` directory if
you want a slimmer checkout — about **143 MB** of weight lives here.

## What's where

| Subdir                               | Size   | What it is                                                                  |
|--------------------------------------|--------|-----------------------------------------------------------------------------|
| `garbage/`                           | ~104 MB| Accidental ImageMagick PostScript dumps, pip-typo files, editor crash files, committed binary |
| `recorded_data/`                     | ~42 MB | `spectrum_data.txt`, `alignment.json`, `spectrum_output.mid` — outputs of past runs |
| `duplicate_misnamed_top_level/`      | small  | 15 top-level files whose names lied about their contents (all identical to base viz) |
| `originals_backups_minimal/`         | small  | The original `backups_minimal_20251016_235901/` dir untouched                |
| `originals_consolidated_code/`       | small  | The original `consolidated_code/` dir untouched                              |
| `originals_top_level/`               | small  | A few stray top-level originals not yet categorized                          |

Each subdir has its own `README.md` explaining provenance.

## Why keep all this

You asked for a conservative reorg. Nothing was deleted — duplicates and junk were just
quarantined here so the live tree stays clean. If after a few weeks you haven't needed
anything from this directory, `rm -rf archive/` is safe.
