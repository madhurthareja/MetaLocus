"""Edge -> stereo audio cue. No live playback device here -- this renders
to an in-memory array (and optionally a .wav file), which is all a
headless DGX Spark or a test runner needs; a real device sink is a
different, much smaller module layered on top of this one.

Two things encode the geometry of an Edge into sound:
- stereo pan encodes bearing (straight ahead = centered, left/right of
  travel = panned that way), clamped to +/-90 degrees since you can't
  pan further than hard left/right.
- an amplitude pulse encodes time-to-collision: the closer the hazard,
  the faster the pulse. No TTC (edge.ttc_s is None, i.e. not closing) is
  a steady, unpulsed tone.
"""

from __future__ import annotations

import numpy as np

from locus.types import Edge

DEFAULT_SAMPLE_RATE = 44100


def render_cue(
    edge: Edge,
    duration_s: float = 0.3,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    base_freq_hz: float = 440.0,
    min_ttc_s: float = 0.3,
    max_pulse_hz: float = 8.0,
) -> np.ndarray:
    """Returns a (n_samples, 2) float32 array in [-1, 1]."""
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    tone = np.sin(2 * np.pi * base_freq_hz * t)

    if edge.ttc_s is not None and edge.ttc_s > 0:
        pulse_hz = min(max_pulse_hz, 1.0 / max(edge.ttc_s, min_ttc_s))
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * pulse_hz * t - np.pi / 2))
    else:
        envelope = np.ones(n)

    mono = tone * envelope

    pan_rad = float(np.clip(edge.bearing_rad, -np.pi / 2, np.pi / 2))
    pan_t = (pan_rad + np.pi / 2) / np.pi  # 0 = full left, 1 = full right
    left_gain = np.cos(pan_t * np.pi / 2)
    right_gain = np.sin(pan_t * np.pi / 2)  # equal-power panning

    stereo = np.stack([mono * left_gain, mono * right_gain], axis=1)
    return stereo.astype(np.float32)


def write_cue(
    path: str,
    edge: Edge,
    duration_s: float = 0.3,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    **kwargs,
) -> None:
    import soundfile as sf

    audio = render_cue(edge, duration_s=duration_s, sample_rate=sample_rate, **kwargs)
    sf.write(path, audio, sample_rate)
