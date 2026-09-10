"""A server that checks an env client against SPEC.md and says what it broke.

`plugrl-server` is the real thing, and it is forgiving: it validates the
`message_type` and the presence of a few keys, then hands everything else to
the policy. That is the right trade for a training run and the wrong one for
someone writing a client in a new language, who wants to be told which
clause they violated rather than to watch the loss fail to move.

This server speaks the same protocol and checks every clause of SPEC.md that
can be checked from one side of the wire. It prints a report and exits
non-zero if anything failed, so it can sit in a CI job.

    python conformance_server.py --port 8000 --steps 20
    ./plugrl_client 127.0.0.1 8000 20        # in another terminal

What it deliberately does NOT require, because SPEC.md does not:

  * that the feedback's env set matches the infer's (section 4.3);
  * that `n` stays the same between requests (section 5.2);
  * any particular key in the observation, or any camera at all;
  * the `<U` form of the text field, since section 3.4 records that the two
    shipped clients disagree and that both are accepted today.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

import numpy as np
import websockets.asyncio.server as ws_server

# Use the real codec, not a re-implementation: the point is to test a client
# against what plugrl-server actually does.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from plugrl_protocol import msgpack_numpy  # noqa: E402
from plugrl_protocol.websocket_protocol import MessageType  # noqa: E402


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

    def ok(self, clause: str) -> None:
        self.checked[clause] = self.checked.get(clause, 0) + 1

    def fail(self, clause: str, detail: str) -> None:
        self.checked.setdefault(clause, 0)
        self.failures.append((clause, detail))

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
            self.advisories.setdefault(clause, detail)

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


class ConformanceServer:
    def __init__(self, horizon: int, action_dim: int, action_dtype: str, steps: int):
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_dtype = np.dtype(action_dtype)
        self.steps = steps
        self.report = Report()
        self.exchanges = 0
        self.finished = asyncio.Event()

    async def handle(self, websocket) -> None:
        report = self.report
        packer = msgpack_numpy.Packer()

        # Section 5.1: the server speaks first, with an envelope. Everything
        # after this point is the client's turn to be judged.
        await websocket.send(
            packer.pack({"message_type": str(MessageType.METADATA), "data": {}})
        )

        awaiting = "infer"
        try:
            while self.exchanges < self.steps:
                payload = msgpack_numpy.unpackb(await websocket.recv())

                if not isinstance(payload, dict):
                    report.fail("2 message is a map", f"got {type(payload).__name__}")
                    return
                report.ok("2 message is a map")

                kind = payload.get("message_type")
                report.require(
                    isinstance(kind, str),
                    "2 message_type is a string, not bytes",
                    f"got {type(kind).__name__}",
                )
                if not report.require(
                    kind == awaiting,
                    "4.2 infer and feedback strictly alternate",
                    f"expected {awaiting!r}, received {kind!r}",
                ):
                    return

                if kind == str(MessageType.INFER):
                    n = self.check_infer(payload)
                    if n is None:
                        return
                    await websocket.send(packer.pack(self.action_for(payload, n)))
                    awaiting = str(MessageType.FEEDBACK)
                else:
                    self.check_feedback(payload)
                    awaiting = str(MessageType.INFER)
                    self.exchanges += 1
        except Exception as exc:  # noqa: BLE001 - the report is the output
            report.fail("connection", f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.set()

    def check_infer(self, payload: dict) -> int | None:
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


async def serve(args: argparse.Namespace) -> int:
    server = ConformanceServer(
        horizon=args.horizon,
        action_dim=args.action_dim,
        action_dtype=args.action_dtype,
        steps=args.steps,
    )
    async with ws_server.serve(
        server.handle, args.host, args.port, compression=None, max_size=None
    ):
        print(
            f"conformance server on ws://{args.host}:{args.port} - "
            f"expecting {args.steps} exchanges, horizon {args.horizon}, "
            f"action dtype {args.action_dtype}",
            flush=True,
        )
        try:
            await asyncio.wait_for(server.finished.wait(), timeout=args.timeout)
        except asyncio.TimeoutError:
            server.report.fail("connection", f"no client finished in {args.timeout}s")

    print(f"\ncompleted {server.exchanges} infer/action/feedback exchanges")
    print(server.report.render())
    return 1 if server.report.failures else 0


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
    return asyncio.run(serve(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
