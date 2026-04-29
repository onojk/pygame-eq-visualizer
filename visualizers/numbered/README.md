
## visualizer 13 — live plasma warp

13_video_warp.py uses live PipeWire monitor capture via `pw-record`.
Press **H** to toggle the debug overlay.

## System requirements (Linux/PipeWire)

On systems with SOF audio hardware (most modern Intel laptops), this
visualizer needs PipeWire's quantum increased to prevent buffer
underruns during sustained capture. Create:

    ~/.config/pipewire/pipewire.conf.d/99-big-buffers.conf

with these contents:

    context.properties = {
        default.clock.rate          = 48000
        default.clock.allowed-rates = [ 48000 ]
        default.clock.quantum       = 1024
        default.clock.min-quantum   = 512
        default.clock.max-quantum   = 2048
    }

Then restart the audio stack:

    systemctl --user restart pipewire pipewire-pulse wireplumber

Without this config, the visualizer may stall every 30-60 seconds
with audible playback hiccups during recovery.
