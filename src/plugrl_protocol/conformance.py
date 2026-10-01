"""A server that checks an env client against SPEC.md and says what it broke.

`plugrl-server` is the real thing, and it is forgiving: it validates the
`message_type` and the presence of a few keys, then hands everything else to
the policy. That is the right trade for a training run and the wrong one for
someone writing a client in a new language, who wants to be told which
clause they violated rather than to watch the loss fail to move.

This server speaks the same protocol and checks the clauses of SPEC.md that
can be checked from the server's side of the wire. It prints a report and
exits non-zero if anything failed, so it can sit in a CI job.

One command, which starts your client once the server is listening and waits
for both:

    plugrl-conformance --steps 20 --client "./plugrl_client 127.0.0.1 8000 20"

Or the two halves separately, which is what `examples/conformance_server.py`
has always done:

    plugrl-conformance --port 8000 --steps 20 &
    ./plugrl_client 127.0.0.1 8000 20

The first form exists because the second has a race in it: a client started
before the server is listening fails for a reason that has nothing to do with
the specification.

Two modes. By default the server watches one well-behaved connection, which
is all it can do with a client running an environment it knows nothing about.
With `--probe` the client runs the probe environment of SPEC.md section 8.1
instead. Its observation says how many steps an episode has taken and which
action was applied last, so the server can tell what the client did with each
action chunk: whether it summed the reward, sent the terminal observation,
applied the actions time-major and in order, and read the dtype. Probe mode
also drives the connection rather than watching it - it speaks late, sends an
oversized metadata frame, and with `--scenario` closes the connection for a
resync, stops the run, or sends a text frame - so the clauses about closing
and reconnecting are checked too:

    plugrl-conformance --probe --scenario all \
        --client "python examples/raw_client.py --probe --batch 3"

What it deliberately does NOT require, because SPEC.md does not:

  * that the feedback's env set matches the infer's (section 4.3);
  * that `n` stays the same between requests (section 5.2);
  * any particular key in the observation, or any camera at all, outside
    probe mode;
  * the `<U` form of the text field, since section 3.4 records that the two
    shipped clients disagree and that both are accepted today.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import websockets.asyncio.server as ws_server
import websockets.exceptions as ws_exceptions

# The real codec, not a re-implementation: the point is to check a client
# against what plugrl-server actually does. This module lives in the package
# for that reason, and because a checker worth running should be installable
# rather than a path into somebody's checkout.
from plugrl_protocol import msgpack_numpy
from plugrl_protocol.reuse import REUSE_FEEDBACK_OBS, ObservationCache, ReuseError
from plugrl_protocol.websocket_protocol import (
    SERVER_RESYNC_REASON,
    SERVER_STOP_REASON,
    MessageType,
)

# Probe mode waits this long after the handshake before it speaks, so that a
# client which sends before reading the metadata is caught doing it.
METADATA_DELAY = 0.3
# Larger than the 1 MiB frame cap websockets libraries default to, so a
# client that kept the cap fails to receive the metadata at all.
METADATA_PADDING = 2 * 1024 * 1024
RECONNECT_TIMEOUT = 15.0
CLIENT_EXIT_TIMEOUT = 30.0
SCENARIOS = ("basic", "resync", "text")


class Report:
    """Which clauses were exercised, which were violated, and what was odd.

    Two severities, and the distinction is the point. A **violation** is
    something `plugrl-server` would reject or mis-handle. An **advisory** is
    something it accepts but that differs from what `plugrl-env-client`
    sends - a portability risk, not a breach, and not a reason to fail a CI
    job. Reporting the second as the first would mean enforcing rules the
    protocol does not actually have, which is how a specification stops
    describing its implementation.
    """

    def __init__(self):
        self.checked: dict[str, int] = {}
        self.failures: list[tuple[str, str]] = []
        self.advisories: dict[str, str] = {}
        # Prefixed to every detail, so a report covering several scenarios
        # says which one a failure came from.
        self.context = ""

    def _where(self, detail: str) -> str:
        return f"[{self.context}] {detail}" if self.context else detail

    def ok(self, clause: str) -> None:
        self.checked[clause] = self.checked.get(clause, 0) + 1

    def fail(self, clause: str, detail: str) -> None:
        self.checked.setdefault(clause, 0)
        self.failures.append((clause, self._where(detail)))

    def require(self, condition: bool, clause: str, detail: str) -> bool:
        if condition:
            self.ok(clause)
        else:
            self.fail(clause, detail)
        return condition

    def advise(self, condition: bool, clause: str, detail: str) -> None:
        """Accepted, but worth saying out loud. Never fails the run."""
        self.ok(clause)
        if not condition:
            self.advisories.setdefault(clause, self._where(detail))

    def render(self) -> str:
        lines = ["", "=" * 68, "SPEC.md conformance report", "=" * 68]
        for clause in sorted(self.checked):
            broke = [d for c, d in self.failures if c == clause]
            if broke:
                mark = "FAIL"
            elif clause in self.advisories:
                mark = "note"
            else:
                mark = "ok  "
            lines.append(f"  {mark}  {clause}  ({self.checked[clause]} checks)")
            for detail in broke[:3]:
                lines.append(f"          {detail}")
            if not broke and clause in self.advisories:
                lines.append(f"          {self.advisories[clause]}")
        lines.append("=" * 68)
        summary = (
            f"{len(self.failures)} violation(s)" if self.failures else "no violations"
        )
        if self.advisories:
            summary += f", {len(self.advisories)} advisory note(s)"
        lines.append(summary)
        return "\n".join(lines)


def _is_int_vector(value) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind == "i"


def _batch_sizes(observation: dict) -> set[int]:
    """Leading dimensions of every array under images and states."""
    sizes = set()
    for group in ("images", "states"):
        for array in (observation.get(group) or {}).values():
            if isinstance(array, np.ndarray) and array.ndim >= 1:
                sizes.add(int(array.shape[0]))
    text = observation.get("text")
    if isinstance(text, np.ndarray) and text.ndim == 1:
        sizes.add(int(text.shape[0]))
    elif isinstance(text, (list, tuple)):
        sizes.add(len(text))
    return sizes


def check_observation(report: Report, observation, where: str, expect_n: int) -> None:
    if not report.require(
        isinstance(observation, dict),
        "5.2 observation is a map",
        f"{where}: got {type(observation).__name__}",
    ):
        return

    report.require(
        set(observation) == {"images", "states", "text"},
        "5.2 observation has exactly images, states and text",
        f"{where}: keys were {sorted(observation)}",
    )

    for group in ("images", "states"):
        value = observation.get(group)
        report.require(
            isinstance(value, dict),
            f"5.2 {group} is a map of named arrays",
            f"{where}: {group} was {type(value).__name__}",
        )

    for name, image in (observation.get("images") or {}).items():
        report.require(
            isinstance(image, np.ndarray) and image.dtype == np.uint8,
            "5.2 images are uint8",
            f"{where}: images[{name!r}] dtype "
            f"{getattr(image, 'dtype', type(image).__name__)}",
        )
        report.require(
            isinstance(image, np.ndarray) and image.ndim == 4,
            "5.2 images are [n, h, w, c]",
            f"{where}: images[{name!r}] shape {getattr(image, 'shape', None)}",
        )

    # Section 3.4 Gap: the two shipped clients disagree here and the server
    # accepts both, so this is reported, never failed.
    text = observation.get("text")
    report.advise(
        isinstance(text, np.ndarray) and text.dtype.kind == "U",
        "3.4 text is a numpy <U array, as plugrl-env-client sends",
        f"{where}: text arrived as a {type(text).__name__}; accepted, but a "
        "policy written against the array form would see something else",
    )

    sizes = _batch_sizes(observation)
    report.require(
        len(sizes) <= 1,
        "5.2 every array shares one leading dimension",
        f"{where}: leading dimensions were {sorted(sizes)}",
    )
    if sizes:
        report.require(
            sizes == {expect_n},
            "5.2 observation batch matches env_indices",
            f"{where}: observation batch {sorted(sizes)}, env_indices {expect_n}",
        )


# ------------------------------------------------------------- probe mode
#
# SPEC.md section 8.1. Each probe env counts the steps of its episode in
# `states["t"]` and echoes the action it applied last in `states["a"]`; every
# step pays a reward of 1, and env `i`'s episode ends, terminated, when `t`
# reaches `probe_episode_length(i)`. The server makes every action value say
# where in the chunk it was, so the echo shows exactly which one was applied.


def probe_episode_length(env: int) -> int:
    """3, 5, 7, 3, ...: so chunks end at different steps in different envs."""
    return 3 + 2 * (env % 3)


def probe_action_value(step: int, env: int, dim: int) -> float:
    """10000 per step of the chunk, 10 per env index, 1 per action dimension."""
    return 10000.0 * step + 10.0 * env + dim


def _describe_probe_value(value: float) -> str:
    v = int(round(value))
    return f"step {v // 10000}, env {(v % 10000) // 10}, dim {v % 10}"


class ProbeTracker:
    """What one connection's probe envs must look like, given what was sent."""

    def __init__(self, report: Report, horizon: int, action_dim: int, fresh: bool):
        self.report = report
        self.horizon = horizon
        self.action_dim = action_dim
        # Env indices are connection-scoped (section 4.4), but the client's
        # envs are not: after a reconnect they carry on mid-episode. So only
        # the first connection can expect every env to start at t = 0.
        self.fresh = fresh
        self.state: dict[int, tuple[float, bool]] = {}
        self.pending: dict[int, tuple[float, np.ndarray | None]] = {}

    def _states(self, observation, where: str, n: int):
        report = self.report
        states = observation.get("states") if isinstance(observation, dict) else None
        t = states.get("t") if isinstance(states, dict) else None
        a = states.get("a") if isinstance(states, dict) else None
        if not report.require(
            isinstance(t, np.ndarray) and isinstance(a, np.ndarray),
            "8.1 probe observation carries states t and a",
            f"{where}: states were "
            f"{sorted(states) if isinstance(states, dict) else type(states).__name__}",
        ):
            return None
        if not report.require(
            t.shape == (n, 1) and a.ndim == 2 and a.shape[0] == n,
            "8.1 probe t is [n, 1] and a is [n, d]",
            f"{where}: t {t.shape}, a {a.shape}, n={n}",
        ):
            return None
        return t[:, 0].astype(np.float64), a.astype(np.float64)

    def on_infer(self, env_indices: np.ndarray, observation) -> None:
        report = self.report
        parsed = self._states(observation, "infer", len(env_indices))
        if parsed is None:
            return
        t, _ = parsed
        for row, env in enumerate(env_indices.tolist()):
            report.require(
                env not in self.pending,
                "8 exactly one feedback for every action received",
                f"env {env} asked for a chunk while still holding the last one",
            )
            known = self.state.get(env)
            if known is None:
                if self.fresh:
                    report.require(
                        t[row] == 0,
                        "8.1 a probe env starts its first episode at t = 0",
                        f"env {env} began at t={t[row]:g}",
                    )
            else:
                last_t, terminal = known
                if terminal:
                    report.require(
                        t[row] == 0,
                        "8.1 after a terminal step, the env's next infer starts a new episode",
                        f"env {env} ended at t={last_t:g}, then asked from t={t[row]:g}",
                    )
                else:
                    report.require(
                        t[row] == last_t,
                        "8.1 an infer carries the observation the last feedback reported",
                        f"env {env}: feedback said t={last_t:g}, infer said t={t[row]:g}",
                    )
            self.pending[env] = (float(t[row]), None)

    def action(self, env_indices: np.ndarray, dtype: np.dtype) -> np.ndarray:
        envs = env_indices.tolist()
        action = np.zeros((self.horizon, len(envs), self.action_dim), dtype)
        for row, env in enumerate(envs):
            for k in range(self.horizon):
                for d in range(self.action_dim):
                    action[k, row, d] = probe_action_value(k, env, d)
            start, _ = self.pending.get(env, (0.0, None))
            self.pending[env] = (start, action[:, row, :].astype(np.float64))
        return action

    def on_feedback(self, env_indices: np.ndarray, data) -> None:
        report = self.report
        n = len(env_indices)
        parsed = self._states(
            data.get("obs") if isinstance(data, dict) else None, "feedback", n
        )
        if parsed is None:
            return
        t, a = parsed
        rewards = data.get("rewards")
        terminated = data.get("terminated")
        truncated = data.get("truncated")
        shapes_ok = all(
            isinstance(x, np.ndarray) and x.shape == (n,)
            for x in (rewards, terminated, truncated)
        )
        if not shapes_ok:
            return  # check_feedback has already said why
        for row, env in enumerate(env_indices.tolist()):
            holding = env in self.pending and self.pending[env][1] is not None
            if not holding and not self.fresh and env not in self.state:
                # Never given an action on this connection, so the chunk it
                # is feeding back came from an earlier one.
                report.fail(
                    "7.6 no feedback for an action that arrived on an earlier connection",
                    f"env {env} sent feedback on a reconnected connection that "
                    "never gave it an action",
                )
                continue
            if not report.require(
                holding,
                "4.3 feedback names only envs that are holding an action",
                f"env {env} had no action outstanding",
            ):
                continue
            start, chunk = self.pending.pop(env)
            end = float(t[row])
            length = probe_episode_length(env)
            done = bool(terminated[row])

            report.require(
                not bool(truncated[row]),
                "8.1 the probe env never truncates",
                f"env {env} reported truncated",
            )
            if done and not report.require(
                end == length,
                "5.4 a done step reports the observation that step returned",
                f"env {env} terminated, but its observation says t={end:g}; its "
                f"episode ends at t={length} - this is the next episode's "
                "observation",
            ):
                self.state[env] = (end, True)
                continue

            steps = end - start
            if not report.require(
                steps.is_integer() and 1 <= steps <= self.horizon,
                "5.3 a client executes between 1 and H steps of a chunk",
                f"env {env}: t went from {start:g} to {end:g} with H={self.horizon}",
            ):
                self.state[env] = (end, done)
                continue
            steps = int(steps)

            reward = float(rewards[row])
            hint = (
                " - the last step's reward, not the sum"
                if reward == 1 and steps > 1
                else ""
            )
            report.require(
                reward == steps,
                "5.4 rewards is the sum over the steps of the chunk",
                f"env {env}: {steps} steps of reward 1, reported {reward:g}{hint}",
            )

            expected = chunk[steps - 1]
            applied = a[row]
            if applied.shape == expected.shape and np.array_equal(applied, expected):
                report.ok(
                    "5.3 the action chunk is applied time-major, in order, as sent"
                )
            else:
                first = applied[0] if applied.size else float("nan")
                report.fail(
                    "5.3 the action chunk is applied time-major, in order, as sent",
                    f"env {env}: after {steps} step(s) it applied {applied[:3].tolist()}, "
                    f"expected step {steps - 1}'s action {expected[:3].tolist()}"
                    + (
                        f" (what it applied decodes as {_describe_probe_value(first)})"
                        if np.isfinite(first) and 0 <= first < 1e7
                        else " (not a value the server sent - check the dtype)"
                    ),
                )

            report.require(
                end <= length,
                "5.4 a chunk stops at the step that ends the episode",
                f"env {env}: its episode ends at t={length}, and it reached t={end:g}",
            )
            report.require(
                done == (end == length),
                "5.4 terminated is set on the terminal step, and only there",
                f"env {env}: t={end:g} of {length}, terminated={done}",
            )
            if done:
                episode = (data.get("info") or {}).get("episode")
                if isinstance(episode, dict) and "r" in episode:
                    r = np.asarray(episode.get("r")).reshape(-1)
                    report.advise(
                        r.size > row and float(r[row]) == length,
                        "5.4 info.episode, when sent, reports the episode's return",
                        f"env {env}: an episode of {length} steps returned {length}, "
                        f"info.episode.r said {r.tolist()}",
                    )
            self.state[env] = (end, done)


# ------------------------------------------------------------- the server


class ConformanceServer:
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        action_dtype: str,
        steps: int,
        *,
        probe: bool = False,
        scenario: str = "basic",
        report: Report | None = None,
        features: tuple[str, ...] = (),
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_dtype = np.dtype(action_dtype)
        self.steps = steps
        self.probe = probe
        self.scenario = scenario
        self.features = tuple(features)
        self.reused_rows = 0
        self.report = report if report is not None else Report()
        self.exchanges = 0
        self.connections = 0
        self.resynced = False
        self.stopped = False
        self.finished = asyncio.Event()

    def metadata(self) -> dict:
        data = {
            "protocol_version": 1,
            "server": "conformance_server.py",
            "action_horizon": self.horizon,
            "action_dim": self.action_dim,
        }
        if self.features:
            data["features"] = list(self.features)
        if self.probe:
            # A key no client knows, and big enough that a client which kept
            # its library's frame cap cannot receive this message. Section
            # 5.1 says to ignore unknown keys, section 1.1 forbids the cap.
            data["conformance_padding"] = "x" * METADATA_PADDING
        return {"message_type": str(MessageType.METADATA), "data": data}

    async def handle(self, websocket) -> None:
        if self.probe:
            await self._handle_probe(websocket)
        else:
            await self._handle_passive(websocket)

    def _envelope(self, payload) -> str | None:
        """The message_type of a well-formed envelope, or None."""
        report = self.report
        if not isinstance(payload, dict):
            report.fail("2 message is a map", f"got {type(payload).__name__}")
            return None
        report.ok("2 message is a map")
        kind = payload.get("message_type")
        report.require(
            isinstance(kind, str),
            "2 message_type is a string, not bytes",
            f"got {type(kind).__name__}",
        )
        return kind

    async def _handle_passive(self, websocket) -> None:
        report = self.report
        packer = msgpack_numpy.Packer()

        # Section 5.1: the server speaks first, with an envelope, and the
        # descriptive keys are how a client learns the action shape without
        # being told out of band. They are sent here so that a client which
        # reads them is exercised, and one which ignores them is proved not
        # to break - section 5.1 requires no key to be present.
        await websocket.send(packer.pack(self.metadata()))

        cache = ObservationCache()
        awaiting = "infer"
        try:
            while self.exchanges < self.steps:
                payload = msgpack_numpy.unpackb(await websocket.recv())
                kind = self._envelope(payload)
                if kind is None:
                    return
                if not report.require(
                    kind == awaiting,
                    "4.2 infer and feedback strictly alternate",
                    f"expected {awaiting!r}, received {kind!r}",
                ):
                    return

                if kind == str(MessageType.INFER):
                    n = self.check_infer(payload, cache)
                    if n is None:
                        return
                    await websocket.send(packer.pack(self.action_for(payload, n)))
                    awaiting = str(MessageType.FEEDBACK)
                else:
                    self.check_feedback(payload)
                    self._remember(cache, payload)
                    awaiting = str(MessageType.INFER)
                    self.exchanges += 1
        except ws_exceptions.ConnectionClosedOK:
            # The client said goodbye and left. SPEC.md section 7.5 allows
            # that at any point, so it is not a failure - a client that has
            # collected the episodes it wanted is finished, and the server
            # does not get to call that a protocol violation.
            report.ok("7.5 a client that stops sends a close frame")
        except ws_exceptions.ConnectionClosedError:
            # It vanished without a close frame. Section 7.5 asks for one,
            # and nothing breaks without it, so this is a note.
            report.advise(
                False,
                "7.5 a client that stops sends a close frame",
                "the connection dropped with no close frame; RFC 6455 asks "
                "for one and SPEC.md section 7.5 repeats the ask",
            )
        except Exception as exc:  # noqa: BLE001 - the report is the output
            report.fail("connection", f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.set()

    async def _handle_probe(self, websocket) -> None:
        report = self.report
        packer = msgpack_numpy.Packer()
        self.connections += 1
        index = self.connections

        if self.stopped:
            report.fail(
                "7.1 a client does not reconnect after plugrl-server-stop",
                f"connection {index} arrived after the server stopped the run",
            )
            await websocket.close(1001, SERVER_STOP_REASON)
            return
        if index > 1 and self.scenario == "resync":
            report.ok("7.2 a client reconnects after plugrl-server-resync")

        request = getattr(websocket, "request", None)
        offered = request.headers.get("Sec-WebSocket-Extensions", "") if request else ""
        report.require(
            "permessage-deflate" not in offered.lower(),
            "1.1 a client does not offer compression",
            f"the handshake offered {offered!r}",
        )

        # Speak late, and see whether the client waits for us.
        first = asyncio.ensure_future(websocket.recv())
        done, _ = await asyncio.wait({first}, timeout=METADATA_DELAY)
        if done:
            if first.exception() is None:
                report.fail(
                    "5.1 a client reads metadata before sending anything",
                    f"a message arrived within {METADATA_DELAY}s of the handshake, "
                    "before the server had spoken",
                )
            else:
                report.fail(
                    "connection", f"closed before metadata: {first.exception()}"
                )
            self.finished.set()
            return
        report.ok("5.1 a client reads metadata before sending anything")
        await websocket.send(packer.pack(self.metadata()))

        tracker = ProbeTracker(report, self.horizon, self.action_dim, fresh=index == 1)
        cache = ObservationCache()
        awaiting = str(MessageType.INFER)
        opening = True
        try:
            while True:
                raw = await first if first is not None else await websocket.recv()
                first = None
                if not report.require(
                    isinstance(raw, bytes),
                    "2 every frame is binary",
                    "the client sent a text frame",
                ):
                    return
                payload = msgpack_numpy.unpackb(raw)
                kind = self._envelope(payload)
                if kind is None:
                    return
                if opening and index > 1:
                    report.require(
                        kind == str(MessageType.INFER),
                        "7.6 a reconnecting client drops held feedback and starts with infer",
                        f"its first message on connection {index} was {kind!r}",
                    )
                opening = False
                if not report.require(
                    kind == awaiting,
                    "4.2 infer and feedback strictly alternate",
                    f"expected {awaiting!r}, received {kind!r}",
                ):
                    return

                if kind == str(MessageType.INFER):
                    if self.check_infer(payload, cache) is None:
                        return
                    tracker.on_infer(payload["env_indices"], payload["data"])
                    if self.scenario == "text":
                        await self._send_text_frame(websocket)
                        return
                    action = tracker.action(payload["env_indices"], self.action_dtype)
                    await websocket.send(
                        packer.pack(
                            {
                                "message_type": str(MessageType.ACTION),
                                "data": {
                                    "env_ids": np.asarray(
                                        payload["env_indices"], dtype=np.int64
                                    ),
                                    "action": action,
                                },
                            }
                        )
                    )
                    awaiting = str(MessageType.FEEDBACK)
                    if (
                        self.scenario == "resync"
                        and not self.resynced
                        and self.exchanges >= self.steps // 2
                    ):
                        # The client now holds a feedback it can never send.
                        self.resynced = True
                        await websocket.close(1001, SERVER_RESYNC_REASON)
                        return
                else:
                    self.check_feedback(payload)
                    self._remember(cache, payload)
                    tracker.on_feedback(payload["env_indices"], payload["data"])
                    awaiting = str(MessageType.INFER)
                    self.exchanges += 1
                    if self.exchanges >= self.steps:
                        self.stopped = True
                        await websocket.close(1001, SERVER_STOP_REASON)
                        self.finished.set()
                        return
        except ws_exceptions.ConnectionClosed as closed:
            code = closed.rcvd.code if closed.rcvd is not None else None
            if code == 1009:
                report.fail(
                    "1.1 a client accepts frames larger than 1 MiB",
                    "it closed with 1009 (message too big) on the metadata, "
                    f"which is {METADATA_PADDING // (1024 * 1024)} MiB",
                )
            else:
                report.fail(
                    "connection",
                    f"the client closed the connection (code {code}) after "
                    f"{self.exchanges} exchanges, before the server ended the run",
                )
            self.finished.set()
        except Exception as exc:  # noqa: BLE001 - the report is the output
            report.fail("connection", f"{type(exc).__name__}: {exc}")
            self.finished.set()

    async def _send_text_frame(self, websocket) -> None:
        """Section 7.4: a text frame where binary was expected is fatal."""
        report = self.report
        await websocket.send(
            "Traceback (most recent call last): this text frame stands for a "
            "server-side error, sent by the conformance checker"
        )
        try:
            extra = await asyncio.wait_for(websocket.recv(), timeout=10)
        except ws_exceptions.ConnectionClosed:
            report.ok("7.4 a client treats a text frame as fatal")
        except asyncio.TimeoutError:
            report.fail(
                "7.4 a client treats a text frame as fatal",
                "the connection was still open 10 s after the text frame",
            )
        else:
            kind = "a message"
            try:
                kind = repr(msgpack_numpy.unpackb(extra).get("message_type"))
            except Exception:  # noqa: BLE001 - only for the detail
                pass
            report.fail(
                "7.4 a client treats a text frame as fatal",
                f"after the text frame it went on and sent {kind}",
            )
        self.finished.set()

    @staticmethod
    def _remember(cache: ObservationCache, payload: dict) -> None:
        """What this feedback lets the next infer reuse (section 10.1)."""
        try:
            data = payload["data"]
            cache.on_feedback(
                payload["env_indices"],
                data["obs"],
                data["terminated"],
                data["truncated"],
            )
        except (KeyError, TypeError):
            pass  # check_feedback has already said what is wrong

    def check_infer(self, payload: dict, cache: ObservationCache) -> int | None:
        report = self.report
        for key in ("data", "env_indices", "step_ids"):
            if not report.require(
                key in payload,
                "5.2 infer carries data, env_indices and step_ids",
                f"missing {key!r}",
            ):
                return None

        env_indices = payload["env_indices"]
        if not report.require(
            _is_int_vector(env_indices),
            "5.2 env_indices is a one-dimensional integer array",
            f"got {getattr(env_indices, 'dtype', type(env_indices).__name__)} "
            f"shape {getattr(env_indices, 'shape', None)}",
        ):
            return None

        n = int(env_indices.shape[0])
        report.require(n > 0, "5.2 an infer names at least one env", "n was 0")
        report.require(
            len(set(env_indices.tolist())) == n,
            "4.4 env indices are unique within a message",
            f"repeated indices in {env_indices.tolist()}",
        )
        report.require(
            _is_int_vector(payload["step_ids"]) and payload["step_ids"].shape[0] == n,
            "5.2 step_ids matches env_indices in length",
            f"step_ids {getattr(payload['step_ids'], 'shape', None)} vs n={n}",
        )
        reuse = payload.get("reuse")
        if reuse is not None:
            # Section 10.1. The full observation replaces the sent rows, so
            # everything downstream sees what the server would see.
            if not report.require(
                REUSE_FEEDBACK_OBS in self.features,
                "10.1 a client sends reuse only when the server offers it",
                "the infer carried reuse, and this server's metadata lists no "
                "such feature",
            ):
                return None
            try:
                payload["data"] = cache.complete(env_indices, payload["data"], reuse)
            except ReuseError as exc:
                report.fail(
                    "10.1 a client reuses only an observation the server holds",
                    str(exc),
                )
                return None
            report.ok("10.1 a client reuses only an observation the server holds")
            self.reused_rows += int(np.asarray(reuse).sum())
        check_observation(report, payload["data"], "infer", n)
        return n

    def action_for(self, payload: dict, n: int) -> dict:
        """Time-major, per section 5.3, with env_ids echoed in order."""
        action = np.zeros((self.horizon, n, self.action_dim), self.action_dtype)
        for t in range(self.horizon):
            action[t, :, 0] = t
        return {
            "message_type": str(MessageType.ACTION),
            "data": {
                "env_ids": np.asarray(payload["env_indices"], dtype=np.int64),
                "action": action,
            },
        }

    def check_feedback(self, payload: dict) -> None:
        report = self.report
        for key in ("data", "env_indices", "step_ids"):
            if not report.require(
                key in payload,
                "5.4 feedback carries data, env_indices and step_ids",
                f"missing {key!r}",
            ):
                return

        env_indices = payload["env_indices"]
        if not report.require(
            _is_int_vector(env_indices),
            "5.4 feedback env_indices is a one-dimensional integer array",
            f"got {type(env_indices).__name__}",
        ):
            return
        m = int(env_indices.shape[0])

        data = payload["data"]
        if not report.require(
            isinstance(data, dict), "5.4 feedback data is a map", f"{type(data)}"
        ):
            return
        report.require(
            set(data) == {"obs", "rewards", "terminated", "truncated", "info"},
            "5.4 feedback data has exactly its five keys",
            f"keys were {sorted(data)}",
        )

        rewards = data.get("rewards")
        if report.require(
            isinstance(rewards, np.ndarray)
            and rewards.dtype.kind == "f"
            and rewards.shape == (m,),
            "5.4 rewards is a float array of length m",
            f"got {getattr(rewards, 'dtype', type(rewards).__name__)} "
            f"shape {getattr(rewards, 'shape', None)}, m={m}",
        ):
            # The server never inspects the width, so a wider reward is
            # accepted rather than rejected - but it is not what the Python
            # client sends, and a policy that reshapes by itemsize would
            # notice.
            report.advise(
                rewards.dtype == np.float32,
                "5.4 rewards is float32, as plugrl-env-client sends",
                f"got {rewards.dtype}; accepted, but float32 is the norm",
            )

        for flag in ("terminated", "truncated"):
            value = data.get(flag)
            report.require(
                isinstance(value, np.ndarray)
                and value.dtype == np.bool_
                and value.shape == (m,),
                f"5.4 {flag} is a bool array of length m",
                f"got {getattr(value, 'dtype', type(value).__name__)} "
                f"shape {getattr(value, 'shape', None)}, m={m}",
            )

        report.require(
            isinstance(data.get("info"), dict),
            "5.4 info is a map",
            f"got {type(data.get('info')).__name__}",
        )
        check_observation(report, data.get("obs"), "feedback", m)


async def run_scenario(args: argparse.Namespace, scenario: str, report: Report) -> int:
    """One server, one client launch. Returns the number of exchanges."""
    report.context = scenario if args.probe and args.scenario == "all" else ""
    server = ConformanceServer(
        horizon=args.horizon,
        action_dim=args.action_dim,
        action_dtype=args.action_dtype,
        steps=args.steps,
        probe=args.probe,
        scenario=scenario,
        report=report,
        features=tuple(args.features),
    )
    async with ws_server.serve(
        server.handle, args.host, args.port, compression=None, max_size=None
    ):
        mode = f"probe mode, scenario {scenario}" if args.probe else "passive mode"
        print(
            f"conformance server on ws://{args.host}:{args.port} ({mode}) - "
            f"expecting {args.steps} exchanges, horizon {args.horizon}, "
            f"action dtype {args.action_dtype}",
            flush=True,
        )

        # Started here rather than by the caller, so it cannot start before
        # the socket is listening. A client that fails to connect fails for a
        # reason that has nothing to do with the specification.
        client = None
        if args.client:
            print(f"starting client: {args.client}", flush=True)
            client = await asyncio.create_subprocess_shell(args.client)

        try:
            await asyncio.wait_for(server.finished.wait(), timeout=args.timeout)
        except asyncio.TimeoutError:
            if scenario == "resync" and server.resynced and server.connections < 2:
                report.fail(
                    "7.2 a client reconnects after plugrl-server-resync",
                    f"no new connection within {args.timeout}s of the resync",
                )
            else:
                report.fail("connection", f"no client finished in {args.timeout}s")

        if client is not None:
            # Still listening, so a client that reconnects after the stop is
            # seen doing it.
            try:
                code = await asyncio.wait_for(
                    client.wait(), timeout=CLIENT_EXIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                client.kill()
                await client.wait()
                report.fail(
                    "client process",
                    f"still running {CLIENT_EXIT_TIMEOUT:.0f}s after the exchanges ended",
                )
            else:
                # A client that crashes on the way out has not passed, however
                # clean the exchanges looked. After a text frame (section 7.4)
                # it is expected to stop with an error.
                if not args.probe:
                    if code != 0:
                        report.fail("client process", f"exited with status {code}")
                elif scenario != "text":
                    report.require(
                        code == 0,
                        "7.1 a client exits cleanly on plugrl-server-stop",
                        f"exited with status {code}",
                    )
        elif args.probe and server.stopped:
            await asyncio.sleep(2)  # long enough to see an immediate reconnect
    if REUSE_FEEDBACK_OBS in server.features and scenario != "text":
        # A feature is optional, so not using it is no violation - but a
        # client offered it and never using it is worth saying.
        report.advise(
            server.reused_rows > 0,
            "10.1 a client offered reuse-feedback-obs uses it",
            "it was offered the feature and sent every observation in full",
        )
    return server.exchanges


async def serve(args: argparse.Namespace) -> int:
    report = Report()
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    exchanges = 0
    for scenario in scenarios:
        exchanges += await run_scenario(args, scenario, report)
    print(f"\ncompleted {exchanges} infer/action/feedback exchanges")
    print(report.render())
    return 1 if report.failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument(
        "--action-dtype",
        default="float32",
        help="the environment's action dtype; a client must read the typestr",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--probe",
        action="store_true",
        help=(
            "the client runs the probe environment of SPEC.md section 8.1, so "
            "chunk sums, terminal observations and action order can be checked"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="basic",
        help=(
            "probe mode only. basic: run, then stop with plugrl-server-stop. "
            "resync: close for a resync halfway, then stop. text: answer an "
            "infer with a text frame. all: each in turn, starting the client "
            "once per scenario"
        ),
    )
    parser.add_argument(
        "--features",
        type=lambda text: [f for f in text.split(",") if f],
        default=[],
        help=(
            "comma-separated optional features of SPEC.md section 10 to offer "
            f"in the metadata; known: {REUSE_FEEDBACK_OBS}"
        ),
    )
    parser.add_argument(
        "--client",
        default=None,
        metavar="COMMAND",
        help=(
            "shell command that runs the client under test. Started once the "
            "server is listening and waited for afterwards, so one command "
            "does the whole check. Without it the server waits for a client "
            "started elsewhere, as it always has."
        ),
    )
    args = parser.parse_args()
    unknown = set(args.features) - {REUSE_FEEDBACK_OBS}
    if unknown:
        parser.error(f"unknown features: {sorted(unknown)}")
    if args.scenario != "basic" and not args.probe:
        parser.error("--scenario needs --probe")
    if args.scenario == "all" and not args.client:
        parser.error(
            "--scenario all starts the client once per scenario: pass --client"
        )
    if args.probe and args.action_dim > 9:
        parser.error(
            "probe mode encodes the action dimension in one digit: --action-dim <= 9"
        )
    return asyncio.run(serve(args))


if __name__ == "__main__":
    raise SystemExit(main())
