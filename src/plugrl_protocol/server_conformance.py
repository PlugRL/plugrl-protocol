"""A client that checks a training server against SPEC.md and says what it broke.

`plugrl-conformance` grades an env client. This is the other half: it
connects to a training server - `plugrl-server`, or any other implementation
of the protocol - drives it the way a client may, and checks every server
clause it can see from the wire:

    plugrl-conformance-server --port 8000 --state-dim 3

It runs in phases, each on connections of its own:

  * exchange: the server speaks first, with a metadata message; every action
    is time-major, `[H, n, *da]`, with `env_ids` equal to the request's
    `env_indices` in order, and agrees with the `action_horizon` and
    `action_dim` the metadata declares. The client then does what SPEC.md
    lets it do: ragged batches, a feedback env set that differs from the
    infer set, terminated episodes, and an observation frame over 1 MiB. The
    server must keep the connection open through all of it.
  * errors: a malformed infer, two infers in a row, and a feedback whose
    `info` cannot be split per environment. Each must close that connection
    with 1001 and `plugrl-server-resync` (sections 4.2, 5.4, 7.2), and the
    server must still accept a new connection afterwards.
  * scoping: two connections at once, both using env index 0 (section 4.4).
  * stop, with `--until-stop`: exchange until the server ends the run, which
    it must do with 1001 and `plugrl-server-stop` (section 7.1).

The observation it sends is states only, `states[--state-key]` of
`--state-dim` float32 values per env, which is what a state-based policy
reads; add `--image-key` for one that needs a camera. A server whose policy
needs anything else cannot be driven by this checker.

A server batches inference across its connections and may wait for every
connected client before it infers, so the phases never leave a connection
idle while another one waits.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import websockets.asyncio.client as ws_client
import websockets.exceptions as ws_exceptions

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.conformance import Report
from plugrl_protocol.reuse import REUSE_FEEDBACK_OBS
from plugrl_protocol.websocket_protocol import (
    SERVER_RESYNC_REASON,
    SERVER_STOP_REASON,
    MessageType,
)

BIG_FRAME_PIXELS = 512  # three envs of 512x512x3 uint8 is 2.25 MiB
# What a failed connection attempt raises: refused, timed out, or a server
# that answered the handshake with an HTTP error, as one shutting down does.
CONNECT_ERRORS = (OSError, asyncio.TimeoutError, ws_exceptions.WebSocketException)


class Probe:
    """One connection to the server under test, with the checks it makes."""

    def __init__(self, args: argparse.Namespace, report: Report):
        self.args = args
        self.report = report
        self.packer = msgpack_numpy.Packer()
        self.ws = None
        self.metadata: dict = {}
        self.rng = np.random.default_rng(0)

    # ---------------------------------------------------------- wire

    async def connect(self) -> bool:
        report = self.report
        self.ws = await ws_client.connect(
            f"ws://{self.args.host}:{self.args.port}",
            compression=None,
            max_size=None,
            open_timeout=self.args.timeout,
        )
        try:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=self.args.timeout)
        except asyncio.TimeoutError:
            report.fail(
                "5.1 the server speaks first, with metadata",
                f"nothing arrived within {self.args.timeout}s of the handshake",
            )
            return False
        if not report.require(
            isinstance(raw, bytes),
            "2 every frame is binary",
            "the metadata was a text frame",
        ):
            return False
        message = msgpack_numpy.unpackb(raw)
        if not report.require(
            isinstance(message, dict)
            and message.get("message_type") == str(MessageType.METADATA)
            and isinstance(message.get("data"), dict),
            "5.1 the server speaks first, with metadata",
            f"the first message was {str(message)[:120]}",
        ):
            return False
        self.metadata = message["data"]
        version = self.metadata.get("protocol_version")
        if version is not None:
            report.require(
                version == 1,
                "10 protocol_version, when sent, is 1",
                f"protocol_version was {version!r}",
            )
        features = self.metadata.get("features")
        if features is not None:
            report.require(
                isinstance(features, list)
                and all(isinstance(f, str) for f in features),
                "5.1 features, when sent, is a list of strings",
                f"features was {features!r}",
            )
        return True

    def offers(self, feature: str) -> bool:
        features = self.metadata.get("features")
        return isinstance(features, list) and feature in features

    def observation(self, n: int, *, big: bool = False) -> dict:
        args = self.args
        states = {
            args.state_key: self.rng.normal(size=(n, args.state_dim)).astype(np.float32)
        }
        images = {}
        if args.image_key:
            side = args.image_size
            images[args.image_key] = np.zeros((n, side, side, 3), dtype=np.uint8)
        if big:
            images["conformance_padding"] = np.zeros(
                (n, BIG_FRAME_PIXELS, BIG_FRAME_PIXELS, 3), dtype=np.uint8
            )
        return {"images": images, "states": states, "text": np.asarray(["probe"] * n)}

    def infer_message(
        self, envs: list[int], *, big: bool = False, reuse: list[bool] | None = None
    ) -> bytes:
        message = {
            "message_type": str(MessageType.INFER),
            # With reuse, only the rows the server does not already hold.
            "data": self.observation(len(envs) - sum(reuse or []), big=big),
            "env_indices": np.asarray(envs, dtype=np.int64),
            "step_ids": np.zeros(len(envs), dtype=np.int64),
        }
        if reuse is not None:
            message["reuse"] = np.asarray(reuse, dtype=np.bool_)
        return self.packer.pack(message)

    async def infer(
        self, envs: list[int], *, big: bool = False, reuse: list[bool] | None = None
    ) -> np.ndarray | None:
        """Send an infer and check the action that comes back."""
        report = self.report
        await self.ws.send(self.infer_message(envs, big=big, reuse=reuse))
        raw = await self.recv()
        if raw is None:
            return None
        message = msgpack_numpy.unpackb(raw)
        if not report.require(
            isinstance(message, dict)
            and message.get("message_type") == str(MessageType.ACTION),
            "5.3 an infer is answered with an action",
            f"got {str(message)[:120]}",
        ):
            return None
        data = message.get("data") or {}
        action = data.get("action")
        if not report.require(
            isinstance(action, np.ndarray) and action.ndim >= 2,
            "5.3 action is an array of at least [H, n]",
            f"action was {type(action).__name__} {getattr(action, 'shape', None)}",
        ):
            return None
        report.require(
            action.dtype.kind in "fiu",
            "5.3 action has a numeric dtype",
            f"dtype {action.dtype}",
        )
        horizon, rows = action.shape[0], action.shape[1]
        report.require(
            rows == len(envs),
            "5.3 action is time-major, [H, n, *da]",
            f"{len(envs)} envs asked, action shape {action.shape}",
        )
        report.require(horizon >= 1, "5.3 the horizon is at least 1", f"H = {horizon}")
        env_ids = data.get("env_ids")
        report.require(
            isinstance(env_ids, np.ndarray) and env_ids.tolist() == list(envs),
            "4.3 action env_ids equal the infer's env_indices, in order",
            f"asked {list(envs)}, env_ids "
            f"{env_ids.tolist() if isinstance(env_ids, np.ndarray) else env_ids!r}",
        )
        declared_h = self.metadata.get("action_horizon")
        if isinstance(declared_h, int):
            report.require(
                horizon == declared_h,
                "5.1 the action matches the metadata's action_horizon",
                f"metadata said {declared_h}, action has H = {horizon}",
            )
        declared_d = self.metadata.get("action_dim")
        if isinstance(declared_d, int):
            width = int(np.prod(action.shape[2:])) if action.ndim > 2 else 1
            report.require(
                width == declared_d,
                "5.1 the action matches the metadata's action_dim",
                f"metadata said {declared_d}, action has {width} per env",
            )
        return action

    async def feedback(
        self, envs: list[int], *, done: list[bool] | None = None, info=None
    ):
        done = done or [False] * len(envs)
        if info is None:
            info = {}
            if any(done):
                info = {
                    "episode": {
                        "r": np.asarray([5.0 if d else 0.0 for d in done], np.float32),
                        "l": np.asarray([5 if d else 0 for d in done], np.int64),
                        "s": np.asarray([False] * len(envs)),
                        "mask": np.asarray(done),
                    }
                }
        await self.ws.send(
            self.packer.pack(
                {
                    "message_type": str(MessageType.FEEDBACK),
                    "env_indices": np.asarray(envs, dtype=np.int64),
                    "step_ids": np.zeros(len(envs), dtype=np.int64),
                    "data": {
                        "obs": self.observation(len(envs)),
                        "rewards": np.ones(len(envs), dtype=np.float32),
                        "terminated": np.asarray(done, dtype=np.bool_),
                        "truncated": np.zeros(len(envs), dtype=np.bool_),
                        "info": info,
                    },
                }
            )
        )

    async def recv(self) -> bytes | None:
        """The next binary message, or None with the reason recorded."""
        report = self.report
        try:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=self.args.timeout)
        except asyncio.TimeoutError:
            report.fail("connection", f"no reply within {self.args.timeout}s")
            return None
        except ws_exceptions.ConnectionClosed as closed:
            code = closed.rcvd.code if closed.rcvd else None
            reason = closed.rcvd.reason if closed.rcvd else ""
            report.fail(
                "connection",
                f"the server closed the connection (code {code}, reason {reason!r}) "
                "where SPEC.md lets a client go on",
            )
            return None
        report.require(
            isinstance(raw, bytes), "2 every frame is binary", "got a text frame"
        )
        return raw if isinstance(raw, bytes) else None

    async def expect_resync(self, clause: str, what: str) -> None:
        """The server must close this connection for a resync, and nothing else."""
        report = self.report
        try:
            message = await asyncio.wait_for(self.ws.recv(), timeout=self.args.timeout)
        except ws_exceptions.ConnectionClosed as closed:
            code = closed.rcvd.code if closed.rcvd else None
            reason = closed.rcvd.reason if closed.rcvd else ""
            report.require(
                code == 1001 and reason == SERVER_RESYNC_REASON,
                clause,
                f"after {what}, the server closed with code {code}, reason {reason!r}",
            )
        except asyncio.TimeoutError:
            report.fail(clause, f"after {what}, the connection stayed open")
        else:
            report.fail(
                clause, f"after {what}, the server answered: {str(message)[:80]}"
            )

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()
            self.ws = None


# ---------------------------------------------------------------- phases


async def phase_exchange(args: argparse.Namespace, report: Report) -> None:
    probe = Probe(args, report)
    if not await probe.connect():
        return
    try:
        # All three ask; env 0 finishes first and asks again on its own; the
        # others report afterwards. So the feedback set differs from the
        # infer set, and n varies between infers (sections 4.3 and 5.2).
        if await probe.infer([0, 1, 2]) is None:
            return
        await probe.feedback([0], done=[True])
        if await probe.infer([0]) is None:
            return
        await probe.feedback([1, 2])
        report.ok("4.3 the server accepts a feedback env set unlike the infer's")
        if await probe.infer([2, 1]) is None:  # and in another order
            return
        await probe.feedback([0, 2, 1], done=[False, True, False])
        if await probe.infer([0, 1, 2], big=True) is None:
            return
        report.ok("1.1 the server accepts a frame larger than 1 MiB")
        await probe.feedback([0, 1, 2])
        for _ in range(args.exchanges):
            if await probe.infer([0, 1, 2]) is None:
                return
            await probe.feedback([0, 1, 2])
        report.ok("4.2 the server keeps a well-behaved connection open")
    finally:
        await probe.close()


async def phase_errors(args: argparse.Namespace, report: Report) -> None:
    cases = [
        (
            "7.2 a malformed infer closes the connection for a resync",
            "an infer with no env_indices",
            "malformed",
        ),
        (
            "4.2 two infers in a row close the connection for a resync",
            "a second infer where a feedback was due",
            "double",
        ),
        (
            "5.4 an info that cannot be split per env closes the connection for a resync",
            'a feedback for 2 envs with info {"task": "pick"}',
            "info",
        ),
    ]
    for clause, what, kind in cases:
        probe = Probe(args, report)
        if not await probe.connect():
            return
        try:
            if kind == "malformed":
                await probe.ws.send(
                    probe.packer.pack(
                        {
                            "message_type": str(MessageType.INFER),
                            "data": probe.observation(1),
                            "step_ids": np.zeros(1, dtype=np.int64),
                        }
                    )
                )
            else:
                if await probe.infer([0, 1]) is None:
                    return
                if kind == "double":
                    await probe.ws.send(
                        probe.packer.pack(
                            {
                                "message_type": str(MessageType.INFER),
                                "data": probe.observation(2),
                                "env_indices": np.asarray([0, 1], dtype=np.int64),
                                "step_ids": np.zeros(2, dtype=np.int64),
                            }
                        )
                    )
                else:
                    await probe.feedback([0, 1], info={"task": "pick"})
            await probe.expect_resync(clause, what)
        finally:
            await probe.close()

        # One client's mistake must not end the run for everyone else.
        alive = Probe(args, report)
        try:
            ok = await alive.connect()
        except CONNECT_ERRORS as exc:
            report.fail(
                "7.2 the server survives a client's protocol error",
                f"after {what}, a new connection was refused: {exc}",
            )
            return  # the server is gone; the later cases would only repeat it
        else:
            report.require(
                ok,
                "7.2 the server survives a client's protocol error",
                f"after {what}, a new connection got no metadata",
            )
        finally:
            await alive.close()


async def phase_scoping(args: argparse.Namespace, report: Report) -> None:
    a, b = Probe(args, report), Probe(args, report)
    if not (await a.connect() and await b.connect()):
        await a.close()
        await b.close()
        return
    try:
        for _ in range(3):
            actions = await asyncio.gather(a.infer([0]), b.infer([0]))
            if any(x is None for x in actions):
                return
            await asyncio.gather(a.feedback([0]), b.feedback([0]))
        report.ok("4.4 env indices are connection-scoped: two clients can both use 0")
    finally:
        await a.close()
        await b.close()


async def phase_reuse(args: argparse.Namespace, report: Report) -> None:
    """Section 10.1, when the server offers it."""
    probe = Probe(args, report)
    if not await probe.connect():
        return
    if not probe.offers(REUSE_FEEDBACK_OBS):
        await probe.close()
        report.advise(
            False,
            "10.1 the server offers reuse-feedback-obs",
            "it does not list the feature, so reuse is not checked",
        )
        return
    report.ok("10.1 the server offers reuse-feedback-obs")
    try:
        if await probe.infer([0, 1]) is None:
            return
        await probe.feedback([0, 1], done=[False, True])
        # Env 0 may reuse; env 1's episode ended, so it sends its reset obs.
        if await probe.infer([0, 1], reuse=[True, False]) is None:
            return
        report.ok("10.1 the server answers an infer that reuses observations")
        await probe.feedback([0, 1])
        if await probe.infer([1, 0], reuse=[True, True]) is None:
            return
        report.ok("10.1 the server answers an infer with every row reused")
        await probe.feedback([0, 1], done=[True, False])
        # Env 0 just ended: what it would reuse is its terminal observation.
        await probe.ws.send(probe.infer_message([0, 1], reuse=[True, False]))
        await probe.expect_resync(
            "10.1 a reuse the server cannot honour closes for a resync",
            "an infer reusing env 0's observation right after its episode ended",
        )
    finally:
        await probe.close()

    fresh = Probe(args, report)
    try:
        if not await fresh.connect():
            return
        await fresh.ws.send(fresh.infer_message([0], reuse=[True]))
        await fresh.expect_resync(
            "10.1 a reuse the server cannot honour closes for a resync",
            "an infer reusing the observation of an env never fed back on this connection",
        )
    finally:
        await fresh.close()


async def phase_stop(args: argparse.Namespace, report: Report) -> None:
    probe = Probe(args, report)
    if not await probe.connect():
        return
    try:
        for count in range(args.max_exchanges):
            try:
                await probe.ws.send(
                    probe.packer.pack(
                        {
                            "message_type": str(MessageType.INFER),
                            "data": probe.observation(3),
                            "env_indices": np.asarray([0, 1, 2], dtype=np.int64),
                            "step_ids": np.zeros(3, dtype=np.int64),
                        }
                    )
                )
                await asyncio.wait_for(probe.ws.recv(), timeout=args.timeout)
                await probe.feedback([0, 1, 2], done=[count % 5 == 4] * 3)
            except ws_exceptions.ConnectionClosed as closed:
                code = closed.rcvd.code if closed.rcvd else None
                reason = closed.rcvd.reason if closed.rcvd else ""
                report.require(
                    code == 1001 and reason == SERVER_STOP_REASON,
                    "7.1 the server ends a run with plugrl-server-stop",
                    f"it closed with code {code}, reason {reason!r}",
                )
                return
            except asyncio.TimeoutError:
                report.fail("connection", f"no reply within {args.timeout}s")
                return
        report.fail(
            "7.1 the server ends a run with plugrl-server-stop",
            f"still running after {args.max_exchanges} exchanges; raise --max-exchanges",
        )
    finally:
        await probe.close()


async def run(args: argparse.Namespace) -> int:
    report = Report()
    phases = [
        ("exchange", phase_exchange),
        ("errors", phase_errors),
        ("scoping", phase_scoping),
        ("reuse", phase_reuse),
    ]
    if args.until_stop:
        phases.append(("stop", phase_stop))
    for name, phase in phases:
        report.context = name
        print(f"phase: {name}", flush=True)
        try:
            await phase(args, report)
        except CONNECT_ERRORS as exc:
            report.fail("connection", f"cannot connect to the server: {exc}")
            break
        # Let the server drop the phase's connections before the next one,
        # so it does not wait for a client that has gone.
        await asyncio.sleep(0.5)
    print(report.render())
    return 1 if report.failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state-key", default="obs")
    parser.add_argument("--state-dim", type=int, default=3)
    parser.add_argument("--image-key", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--exchanges", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--until-stop",
        action="store_true",
        help="last, exchange until the server ends the run, and check how it does",
    )
    parser.add_argument("--max-exchanges", type=int, default=2000)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
