# native/

C accelerator used by some of the entropic visualizers.

## Build

```bash
cd native
make
```

That produces `./entropic_worms`. The Python visualizer scripts in
`../visualizers/entropic/` will look for the binary at this location.

## Why this exists

The original repo committed a precompiled `entropic_worms` binary built on someone else's
machine. ELF binaries aren't portable across systems and shouldn't live in source control —
they're now in `archive/garbage/` for reference only. Build from `entropic_worms.c` instead.

## Clean

```bash
make clean
```
