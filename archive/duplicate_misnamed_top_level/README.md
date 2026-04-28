# archive/duplicate_misnamed_top_level/

These 15 files all live at the top level of the original repo and **all have the same MD5
hash** (`d2cdaec7bb730da60adddabd96fac02b`) — they are byte-for-byte identical copies of
`pygamemusicvisualizer.py`.

**The names are misleading.** None of them contain the code their filename suggests:

| Filename                          | What you'd expect             | What it actually is        |
|-----------------------------------|-------------------------------|----------------------------|
| pygamemusicvisualizerNo2..No14.py | numbered variants of the viz  | identical copy of base viz |
| reversevisualszNo1/2/3.py         | reverse audio-video analysis  | identical copy of base viz |
| entropic_visualizer.py            | entropic-engine visualizer    | identical copy of base viz |
| spectrumtomidi.py                 | spectrum → MIDI converter     | identical copy of base viz |

The **real** code for each of these concepts was found in
`backups_minimal_20251016_235901/` (now: `archive/originals_backups_minimal/`). Looks like
someone bulk-copied `pygamemusicvisualizer.py` over all of them by accident — possibly a
glob like `cp pygamemusicvisualizer.py *.py` or a stuck shell loop.

The unique versions of each have been preserved with descriptive names elsewhere:

| Concept                  | Real source                                  | New location                                  |
|--------------------------|----------------------------------------------|-----------------------------------------------|
| numbered variants 2–14   | archive/originals_backups_minimal/*.py       | visualizers/numbered/<descriptive_name>.py    |
| reverse audio-video      | archive/originals_backups_minimal/reverse*.py| visualizers/reverse_analysis/*.py             |
| entropic visualizer      | archive/originals_backups_minimal/entropic*  | visualizers/entropic/entropic_visualizer.py   |
| spectrum to MIDI         | archive/originals_backups_minimal/spectrum*  | tools/spectrumtomidi.py                       |
