"""Writing video a browser will actually play.

OpenCV's `VideoWriter` with the `mp4v` fourcc produces MPEG-4 Part 2. Chrome, Firefox
and Safari all refuse to decode it, so every generation this project wrote was
invisible in the dashboard - the file was on disk, the server returned it with a 200,
and the `<video>` element stayed black. That failure is silent from every angle except
actually watching the page.

PyAV is already a dependency (the DROID loader needs it for AV1), and it ships libx264,
so H.264 in an MP4 container is available with no new install and no ffmpeg binary on
PATH. Three details decide whether the result plays:

  yuv420p        the only chroma format with universal browser support
  even dimensions   H.264 macroblocks are 16x16; libx264 rejects odd width or height
  faststart      moves the moov atom to the front so playback can begin before the
                 whole file has arrived
"""
from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

log = logging.getLogger("penumbra.media")

#: Constant Rate Factor. 20 is visually near-lossless for this content and keeps a
#: 96-frame contact-sheet video comfortably under a megabyte.
CRF = "20"


def write_h264(path: Path | str, frames, fps: float = 15.0, *, crf: str = CRF) -> Path:
    """Write RGB frames to a browser-playable H.264 MP4.

    `frames` is any iterable of `(H, W, 3)` uint8 RGB arrays; they are consumed one at a
    time so a long sequence never has to exist in memory twice.
    """
    path = Path(path)
    it = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError(f"no frames to write to {path}") from None

    h, w = first.shape[:2]
    # libx264 rejects odd dimensions in yuv420p; crop rather than pad so nothing
    # invented appears in a frame the policy is supposed to have seen.
    w -= w % 2
    h -= h % 2

    container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
    try:
        stream = container.add_stream("libx264", rate=Fraction(round(fps * 1000), 1000))
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": crf, "preset": "veryfast"}

        n = 0
        for frame in _chain(first, it):
            arr = np.ascontiguousarray(frame[:h, :w])
            packet = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for encoded in stream.encode(packet):
                container.mux(encoded)
            n += 1
        for encoded in stream.encode():
            container.mux(encoded)
    finally:
        container.close()
    log.debug("wrote %s (%d frames, %dx%d)", path, n, w, h)
    return path


def _chain(first, rest):
    yield first
    yield from rest


def is_browser_playable(path: Path | str) -> bool:
    """Does this file carry a codec a browser will decode?

    Used by the transcode tool to skip files already in H.264, and by tests to catch a
    regression back to mp4v.
    """
    try:
        with av.open(str(path)) as container:
            return any(s.codec_context.name in ("h264", "vp8", "vp9", "av1")
                       for s in container.streams.video)
    except Exception:  # noqa: BLE001
        return False
