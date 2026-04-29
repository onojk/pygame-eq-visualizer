
## visualizer 13 — auto-recovery

13_video_warp.py uses live PipeWire monitor capture via `pw-record`.
On some PipeWire setups the monitor source stalls every ~30–60 seconds;
the visualizer auto-recovers by suspending and resuming the source via
`pactl`, which causes a brief (~1–2 s) audio hiccup. Music continues
across recoveries. Press **H** to toggle the debug overlay.
