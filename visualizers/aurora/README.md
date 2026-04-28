# visualizers/aurora/

Aurora — sunset/atmospheric/painterly visualizers.

| File                                | What it does                                                            |
|-------------------------------------|-------------------------------------------------------------------------|
| `aurora_engine.py`                  | The aurora rendering engine — reusable core                             |
| `aurora_sunset_garden_full.py`      | Full version of the sunset-garden scene (~1000 lines, originally `AuroraSunsetGarden.py`) |
| `aurora_sunset_garden_minimal.py`   | Stripped-down 168-line variant of the same idea (originally `aurora_sunset_garden.py`) |

The two `aurora_sunset_garden_*` scripts are different implementations of the same concept,
not duplicates. The full one has more knobs and effects; the minimal one is easier to read
and modify.
