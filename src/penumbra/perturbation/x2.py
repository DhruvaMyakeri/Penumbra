"""X2 as a perturbation operator over a real episode.

Takes an `Episode`, streams one camera view through `xmax/x2` under a fault's prompt,
and returns a new `Episode` in which only that view's pixels have changed. Physics,
geometry, proprioception, ground-truth actions and the other two camera views are
untouched by construction — they are literally the same arrays.

## The alignment problem, and how it is handled

X2 is a streaming generative model: pushing N frames does not return exactly N. The
smoke test pushed 60 and received 54. Divergence measured against a misaligned stream
would be an artefact of the misalignment, not a finding, so alignment is explicit and
recorded rather than assumed.

1. **Lead-in.** `lead` copies of the first frame are pushed before the episode proper,
   so the model's warm-up (~2.3 s to first output) consumes filler rather than real
   frames. Received frames during that window are discarded by count.
2. **Tail.** `tail` copies of the last frame are pushed afterwards, so the real final
   frames are not the ones lost to drain.
3. **Index mapping.** The remaining M outputs are mapped onto N source positions by
   nearest-neighbour resampling on a linear index map.
4. **Offset refinement.** A small integer shift is searched for, choosing the one that
   maximises mean normalised cross-correlation against the source. The chosen offset
   and its score are written into the result so a reader can judge the alignment.

Step 4 is what makes the geometry gate meaningful: without it, a good perturbation
could be rejected purely for being one frame late.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from collections.abc import Sequence

from ..episodes.types import PERTURBED_VIEW, Episode
from pathlib import Path

from ..config import REPO_ROOT
from ..reactor.session import ReactorSession, TRANSPORT_ERRORS
from .spec import FaultSpec

log = logging.getLogger("penumbra.perturbation")

MODEL_X2 = "xmax/x2"


@dataclass
class PerturbationResult:
    """A perturbed episode plus everything needed to judge and reproduce it."""

    episode: Episode
    fault: str
    fault_version: str
    model: str
    strength: float
    prompt: str
    view: str
    source_frames: int
    received_frames: int
    alignment_offset: int
    alignment_score: float
    lead: int
    tail: int
    seconds: float
    output_shape: tuple[int, int]
    session_id: str | None
    events: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    noop: bool = False
    views: tuple[str, ...] = ()
    pointer: dict | None = None
    reference_image: str | None = None
    per_view_sessions: dict = field(default_factory=dict)
    prime_passes: int = 0

    def to_dict(self) -> dict:
        return {
            "fault": self.fault,
            "fault_version": self.fault_version,
            "model": self.model,
            "strength": self.strength,
            "prompt": self.prompt,
            "view": self.view,
            "source_frames": self.source_frames,
            "received_frames": self.received_frames,
            "alignment_offset": self.alignment_offset,
            "alignment_score": round(self.alignment_score, 4),
            "lead": self.lead,
            "prime_passes": self.prime_passes,
            "tail": self.tail,
            "seconds": round(self.seconds, 2),
            "output_shape": list(self.output_shape),
            "session_id": self.session_id,
            "errors": self.errors,
            "noop": self.noop,
            "views": list(self.views) or [self.view],
            "pointer": self.pointer,
            "reference_image": self.reference_image,
            "per_view_sessions": self.per_view_sessions,
        }


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two grayscale stacks, in [-1, 1]."""
    x = a.astype(np.float32).reshape(len(a), -1)
    y = b.astype(np.float32).reshape(len(b), -1)
    x -= x.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    num = (x * y).sum(axis=1)
    den = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1)) + 1e-8
    return float(np.mean(num / den))


def _gray_small(frames: np.ndarray, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    return np.stack(
        [cv2.cvtColor(cv2.resize(f, size), cv2.COLOR_RGB2GRAY) for f in frames]
    )


def align(source: np.ndarray, received: np.ndarray, max_shift: int = 6) -> tuple[np.ndarray, int, float]:
    """Resample `received` onto `source`'s timeline; return (aligned, offset, score).

    The offset is in *received-stream* frames and is chosen to maximise mean NCC
    against the source, which is what tells us the two streams describe the same
    moments of the same scene.
    """
    n, m = len(source), len(received)
    if m == 0:
        raise RuntimeError("X2 returned no frames")
    h, w = source.shape[1:3]
    resized = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in received])
    src_small = _gray_small(source)

    best = (None, 0, -2.0)
    for offset in range(-max_shift, max_shift + 1):
        idx = np.clip(np.round(np.linspace(0, m - 1, n)).astype(int) + offset, 0, m - 1)
        candidate = resized[idx]
        score = _ncc(src_small, _gray_small(candidate))
        if score > best[2]:
            best = (candidate, offset, score)
    aligned, offset, score = best
    return aligned, offset, score  # type: ignore[return-value]


class X2Perturbation:
    """Streams one view of an episode through X2 under a fault program."""

    model = MODEL_X2

    def __init__(
        self,
        *,
        view: str = PERTURBED_VIEW,
        views: Sequence[str] | None = None,
        lead: int = 24,
        tail: int = 12,
        push_fps: float = 15.0,
        drain_seconds: float = 12.0,
        keep_backlog: bool = True,
        prime_passes: int = 0,
        session_retries: int = 2,
    ) -> None:
        # X2 has exactly one input track, so perturbing several views means several
        # sequential sessions - one per view - not one session with three tracks.
        # Each costs its own connect and its own generative sample; because X2 has no
        # seed, the views are perturbed independently rather than coherently, which is
        # a real property of the intervention and is recorded as such.
        self.views = tuple(views) if views else (view,)
        self.view = self.views[0]
        #: How many times a view is re-attempted when the Reactor session drops. Not a
        #: quality knob - it only covers transport failures.
        self.session_retries = session_retries
        self.lead = lead
        self.tail = tail
        self.push_fps = push_fps
        self.drain_seconds = drain_seconds
        self.keep_backlog = keep_backlog
        #: Extra whole-clip passes pushed before the one that is kept. X2's prompt
        #: applies "from the next block", and a lead-in of a single repeated frame gives
        #: the model no motion to settle against - a run whose prompts were merely
        #: descriptive came back with no visible change in 7 of 18 renders. Priming
        #: streams the real clip first and throws the output away, so the edit is
        #: already committed when the frames that count arrive. Costs a full render per
        #: pass, which is the trade being made.
        self.prime_passes = prime_passes

    async def _apply_one_resilient(
        self, episode: Episode, fault: FaultSpec, strength: float, view: str, *,
        run_id: str,
    ) -> PerturbationResult:
        """One view, retrying when the Reactor session drops underneath us.

        Measured three times in this project: a session reports ready, then leaves the
        ready state mid-stream, and every subsequent `push_frame` raises INVALID_STATE
        because a publish does not survive a reconnect. Twice that killed a run outright
        - once a whole head-to-head, once a 21-situation suite eight minutes in, after
        the director work was already paid for.

        A dropped connection is not a property of the situation being tested, so it must
        not end the experiment. Retrying is also cheap in the sense that matters: X2
        takes no seed, so a re-render was never the *same* render anyway - every attempt
        is a fresh draw from the same distribution, which is exactly what the rest of
        the pipeline already assumes.

        Only transport failures are retried. A gate rejection or an empty render is a
        result and is returned untouched.
        """
        last: Exception | None = None
        for attempt in range(1, self.session_retries + 2):
            try:
                return await self._apply_one(episode, fault, strength, view,
                                             run_id=run_id if attempt == 1
                                             else f"{run_id}-r{attempt}")
            except TRANSPORT_ERRORS as exc:
                last = exc
                if attempt > self.session_retries:
                    break
                delay = 4.0 * attempt
                log.warning("%s on %s: %s - reconnecting in %.0fs (attempt %d of %d)",
                            type(exc).__name__, view, str(exc)[:120], delay,
                            attempt + 1, self.session_retries + 1)
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"x2: {view} failed after {self.session_retries + 1} session attempts; "
            f"last error {type(last).__name__}: {last}"
        ) from last

    async def apply(
        self, episode: Episode, fault: FaultSpec, strength: float, *, run_id: str = "0"
    ) -> PerturbationResult:
        """Transform every configured view, one X2 session each."""
        if len(self.views) == 1:
            return await self._apply_one_resilient(episode, fault, strength, self.view,
                                                   run_id=run_id)

        result: PerturbationResult | None = None
        current = episode
        sessions: dict = {}
        total_seconds = 0.0
        for view in self.views:
            result = await self._apply_one_resilient(current, fault, strength,
                                                        view, run_id=run_id)
            current = result.episode
            sessions[view] = {
                "session_id": result.session_id,
                "received_frames": result.received_frames,
                "alignment_offset": result.alignment_offset,
                "alignment_score": round(result.alignment_score, 4),
                "seconds": round(result.seconds, 2),
            }
            total_seconds += result.seconds
        assert result is not None
        result.episode = current
        result.views = self.views
        result.per_view_sessions = sessions
        result.seconds = total_seconds
        result.episode.episode_id = (
            f"{episode.episode_id}#{fault.name}[{len(self.views)}views]@{strength:g}r{run_id}"
        )
        return result

    async def _apply_one(
        self,
        episode: Episode,
        fault: FaultSpec,
        strength: float,
        view: str,
        *,
        run_id: str = "0",
    ) -> PerturbationResult:
        rung = fault.rung_at(strength)
        source = episode.frames[view]

        if fault.is_noop(strength):
            # The zero rung is the untouched control. It must not touch Reactor at
            # all: routing it through the model would perturb it by definition.
            return PerturbationResult(
                episode=episode,
                fault=fault.name,
                fault_version=fault.version,
                model="none",
                strength=strength,
                prompt="",
                view=view,
                views=(view,),
                source_frames=len(source),
                received_frames=len(source),
                alignment_offset=0,
                alignment_score=1.0,
                lead=0,
                tail=0,
                seconds=0.0,
                output_shape=(source.shape[1], source.shape[2]),
                session_id=None,
                noop=True,
            )

        received: list[np.ndarray] = []
        events: list[dict] = []
        t_begin = time.time()

        async with ReactorSession(MODEL_X2) as s:
            s.on_event(lambda m: events.append(m))
            out_track = s.track("main_video")

            @out_track.on_frame
            def _on_frame(frame: np.ndarray) -> None:
                received.append(frame.copy())

            await s.command("set_keep_backlog", {"keep_backlog": self.keep_backlog})
            await s.command("set_prompt", {"prompt": rung.prompt})

            if fault.reference_image:
                ref_path = Path(fault.reference_image)
                if not ref_path.is_absolute():
                    ref_path = REPO_ROOT / ref_path
                if not ref_path.exists():
                    raise FileNotFoundError(
                        f"fault {fault.name!r} references {fault.reference_image!r}, "
                        f"which does not exist at {ref_path}"
                    )
                ref = await s.upload(str(ref_path))
                await s.command("set_reference_image", {"reference_image": ref})

            pointer = fault.pointer_for(view)
            if pointer:
                await s.command(
                    "set_pointer",
                    {
                        "x": float(pointer.get("x", 0.5)),
                        "y": float(pointer.get("y", 0.5)),
                        "active": bool(pointer.get("active", False)),
                    },
                )
                log.info("%s: pointer (%.3f, %.3f) active=%s",
                         view, pointer.get("x", 0.5), pointer.get("y", 0.5),
                         pointer.get("active", False))

            track = await s.publish("source")
            stream = np.concatenate(
                [
                    np.repeat(source[:1], self.lead, axis=0),
                    # Priming passes: the real clip, streamed and discarded, so the
                    # model has committed to the edit before the frames that count.
                    *([source] * self.prime_passes),
                    source,
                    np.repeat(source[-1:], self.tail, axis=0),
                ]
            )
            #: Everything before the kept pass is filler and is dropped by count below.
            warmup_pushed = self.lead + self.prime_passes * len(source)
            t0 = time.time()
            period = 1.0 / self.push_fps
            for i, frame in enumerate(stream):
                track.push_frame(np.ascontiguousarray(frame))
                await asyncio.sleep(max(0.0, t0 + (i + 1) * period - time.time()))

            # Drain: stop as soon as output stops growing, rather than always waiting.
            deadline = time.time() + self.drain_seconds
            stable_for = 0.0
            last = -1
            while time.time() < deadline:
                await asyncio.sleep(0.5)
                if len(received) == last:
                    stable_for += 0.5
                    if stable_for >= 2.0 and len(received) >= len(source):
                        break
                else:
                    stable_for = 0.0
                    last = len(received)

            session = s.log

        if not received:
            raise RuntimeError(
                f"X2 returned no frames for fault={fault.name} strength={strength}. "
                f"errors={session.errors}"
            )

        # Discard the warm-up proportionally: outputs are produced at roughly the push
        # rate, so the first warmup*(M/total) frames correspond to filler - the lead-in
        # and any priming passes alike.
        total_pushed = len(stream)
        drop = int(round(len(received) * warmup_pushed / total_pushed))
        body = received[drop:] if len(received) - drop >= max(4, len(source) // 4) else received

        aligned, offset, score = align(source, np.stack(body))
        perturbed = episode.with_view(
            view,
            aligned,
            episode_id=f"{episode.episode_id}#{fault.name}[{view}]@{strength:g}r{run_id}",
        )
        return PerturbationResult(
            episode=perturbed,
            fault=fault.name,
            fault_version=fault.version,
            model=MODEL_X2,
            strength=strength,
            prompt=rung.prompt,
            view=view,
            views=(view,),
            pointer=fault.pointer_for(view),
            reference_image=fault.reference_image,
            source_frames=len(source),
            received_frames=len(received),
            alignment_offset=offset,
            alignment_score=score,
            lead=self.lead,
            tail=self.tail,
            prime_passes=self.prime_passes,
            seconds=time.time() - t_begin,
            output_shape=(int(received[0].shape[0]), int(received[0].shape[1])),
            session_id=session.session_id,
            events=events,
            errors=session.errors,
        )
