
## visualizer 13 — known limitation

13_video_warp.py uses live PipeWire monitor capture via `pw-record`.
On some PipeWire/Ubuntu setups (kernel + sof-soundwire hardware), the
audio system starves playback after 5 seconds to 2 minutes of sustained
capture. The visualizer keeps animating but with stale audio data.

Workaround: Ctrl+C the visualizer to restore audio playback. Restart
to capture again. A future revision should add file-playback mode as
an alternative.

This is a known PipeWire-on-Ubuntu issue, not a bug in the visualizer.
`pw-record` itself works correctly when run alone.
