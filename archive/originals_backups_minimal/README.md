# archive/originals_backups_minimal/

The original `backups_minimal_20251016_235901/` directory, untouched.

This was the **good** version of the numbered series — when the top-level `pygamemusicvisualizer*.py` files got corrupted/duplicated (16 files all became byte-identical to the base), this backup directory still had the unique versions.

The unique scripts from here have been **copied** (not moved) into `visualizers/numbered/` and `visualizers/reverse_analysis/` with descriptive names, so the live project doesn't depend on this directory. It's preserved here purely so you can verify the rescue was correct.

## How to verify

```bash
# Show that visualizers/numbered/12_lightning.py came from here:
md5sum archive/originals_backups_minimal/_files/pygamemusicvisualizerNo12.py \
       visualizers/numbered/12_lightning.py
# Both hashes should match.
```
