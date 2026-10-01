"""A PlugRL training server that trains nothing, written against SPEC.md alone.

The other side of `raw_client.py`. It answers every infer with a random
action chunk and counts the feedback it gets, and that is all - no policy,
no algorithm, nothing from plugrl-server. What it does do is every server
clause of SPEC.md: it speaks first; it answers time-major, with `env_ids`; it
validates what it receives and closes for a resync on anything malformed,
including an `info` that cannot be split per environment; it keeps one
client's mistake from ending the run for the others; and when it has had
`--steps` frames it ends the run with `plugrl-server-stop`.

    python reference_server.py --port 8000 --horizon 4 --action-dim 3 --steps 300

`plugrl-conformance-server` grades it, and `--bug NAME` makes it break one
server clause on purpose, so the grader's tests can show that each one is
checked.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

import numpy as np
import websockets.asyncio.server as ws_server
import websockets.exceptions as ws_exceptions

# So a fresh checkout works without installing anything first.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plugrl_protocol import msgpack_numpy  # noqa: E402
from plugrl_protocol.websocket_protocol import (  # noqa: E402
    SERVER_RESYNC_REASON,
    SERVER_STOP_REASON,
    MessageType,
)

BUGS = (
    "no-metadata",  # waits for the client instead of speaking first
    "env-major",  # sends the action as [n, H, da]
    "no-env-ids",  # leaves env_ids out of the action
    "wrong-horizon",  # declares H in the metadata and sends H + 1
    "lenient",  # accepts a malformed infer instead of closing for a resync
    "crash-on-info",  # dies on an info it cannot split, as plugrl-server did before #108
    "plain-stop",  # ends the run with a plain close, no plugrl-server-stop
)


class ProtocolError(ValueError):
    """The client's message breaks SPEC.md; its connection closes for a resync."""


def _int_vector(value, what: str) -> np.ndarray:
    if not (
        isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind == "i"
    ):
        raise ProtocolError(f"{what} is not a one-dimensional integer array")
    return value


def _leading(observation, what: str) -> int:
    if not isinstance(observation, dict) or set(observation) != {
        "images",
        "states",
        "text",
    }:
        raise ProtocolError(f"{what} is not a map of images, states and text")
    sizes = {
        int(array.shape[0])
        for group in ("images", "states")
        for array in (observation[group] or {}).values()
        if isinstance(array, np.ndarray) and array.ndim >= 1
    }
    if len(sizes) > 1:
        raise ProtocolError(
            f"{what} arrays disagree on their leading dimension: {sorted(sizes)}"
        )
    return sizes.pop() if sizes else -1


def _info_entries(info: dict) -> int | None:
    """How many per-env entries an info splits into: SPEC section 5.4's rule.

    The leading dimension of the first ndarray at the top level or one level
    into a nested map; None if there is none, which only works for m = 1.
    """
    for value in info.values():
        if isinstance(value, np.ndarray):
            return int(value.shape[0])
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, np.ndarray):
                    return int(nested.shape[0])
    return None


class ReferenceServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.frames = 0
        self.stopping = asyncio.Event()
        self.connections: set = set()
        self.rng = np.random.default_rng(0)

    def check_infer(self, message: dict) -> np.ndarray:
        if message.get("message_type") != str(MessageType.INFER):
            raise ProtocolError(f"expected infer, got {message.get('message_type')!r}")
        if self.args.bug == "lenient":
            return np.asarray(message.get("env_indices", [0]), dtype=np.int64)
        for key in ("data", "env_indices", "step_ids"):
            if key not in message:
                raise ProtocolError(f"infer has no {key!r}")
        envs = _int_vector(message["env_indices"], "env_indices")
        n = _leading(message["data"], "the observation")
        if n not in (-1, len(envs)):
            raise ProtocolError(f"observation batch {n}, env_indices {len(envs)}")
        return envs

    def check_feedback(self, message: dict, holding: set) -> list[int]:
        if message.get("message_type") != str(MessageType.FEEDBACK):
            raise ProtocolError(
                f"expected feedback, got {message.get('message_type')!r}"
            )
        for key in ("data", "env_indices", "step_ids"):
            if key not in message:
                raise ProtocolError(f"feedback has no {key!r}")
        envs = _int_vector(message["env_indices"], "env_indices").tolist()
        m = len(envs)
        data = message["data"]
        if not isinstance(data, dict) or set(data) != {
            "obs", "rewards", "terminated", "truncated", "info"
        }:  # fmt: skip
            raise ProtocolError("feedback data does not have exactly its five keys")
        for key in ("rewards", "terminated", "truncated"):
            if not (isinstance(data[key], np.ndarray) and data[key].shape == (m,)):
                raise ProtocolError(f"{key} is not an array of length {m}")
        if _leading(data["obs"], "the feedback observation") not in (-1, m):
            raise ProtocolError("feedback observation batch does not match env_indices")
        info = data["info"]
        if not isinstance(info, dict):
            raise ProtocolError("info is not a map")
        if info:
            entries = _info_entries(info)
            if (entries is None and m > 1) or (entries is not None and entries != m):
                if self.args.bug == "crash-on-info":
                    raise RuntimeError(
                        "could not split info"
                    )  # and takes the server down
                raise ProtocolError(f"info cannot be split into {m} entries")
        # Feedback for an env this connection never answered is a transition
        # with no start: dropped, with a word (SPEC section 7.6 SHOULD).
        stale = [e for e in envs if e not in holding]
        if stale:
            print(
                f"feedback for envs {stale} with no action on this connection; dropped"
            )
        return [e for e in envs if e in holding]

    def action(self, envs: np.ndarray) -> dict:
        horizon = self.args.horizon + (1 if self.args.bug == "wrong-horizon" else 0)
        chunk = self.rng.uniform(-1, 1, (horizon, len(envs), self.args.action_dim))
        if self.args.bug == "env-major":
            chunk = chunk.swapaxes(0, 1)
        data = {"action": chunk.astype(np.float32)}
        if self.args.bug != "no-env-ids":
            data["env_ids"] = np.asarray(envs, dtype=np.int64)
        return {"message_type": str(MessageType.ACTION), "data": data}

    async def handle(self, websocket) -> None:
        packer = msgpack_numpy.Packer()
        self.connections.add(websocket)
        holding: set[int] = set()
        try:
            if self.args.bug != "no-metadata":
                await websocket.send(
                    packer.pack(
                        {
                            "message_type": str(MessageType.METADATA),
                            "data": {
                                "protocol_version": 1,
                                "server": "reference_server.py",
                                "action_horizon": self.args.horizon,
                                "action_dim": self.args.action_dim,
                            },
                        }
                    )
                )
            while True:
                raw = await websocket.recv()
                if not isinstance(raw, bytes):
                    raise ProtocolError("a text frame")
                envs = self.check_infer(msgpack_numpy.unpackb(raw))
                await websocket.send(packer.pack(self.action(envs)))
                holding.update(envs.tolist())
                raw = await websocket.recv()
                if not isinstance(raw, bytes):
                    raise ProtocolError("a text frame")
                for env in self.check_feedback(msgpack_numpy.unpackb(raw), holding):
                    holding.discard(env)
                    self.frames += 1
                if self.frames >= self.args.steps:
                    # serve() closes every connection, this one included,
                    # with the stop reason.
                    self.stopping.set()
                    await websocket.wait_closed()
                    return
        except ProtocolError as exc:
            print(f"protocol error: {exc}; closing for a resync")
            await websocket.close(1001, SERVER_RESYNC_REASON)
        except ws_exceptions.ConnectionClosed:
            pass
        except RuntimeError:
            if self.args.bug == "crash-on-info":
                await websocket.close(1011, "Internal server error.")
                self.stopping.set()  # the whole server goes, as it did
                return
            raise
        finally:
            self.connections.discard(websocket)

    async def serve(self) -> None:
        async with ws_server.serve(
            self.handle, self.args.host, self.args.port, compression=None, max_size=None
        ) as server:
            print(
                f"reference server on ws://{self.args.host}:{self.args.port}",
                flush=True,
            )
            await self.stopping.wait()
            for websocket in list(self.connections):
                if self.args.bug == "plain-stop":
                    await websocket.close()
                else:
                    await websocket.close(1001, SERVER_STOP_REASON)
            server.close()
        print(f"stopped after {self.frames} frames")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--action-dim", type=int, default=3)
    p.add_argument("--steps", type=int, default=300, help="frames before the run ends")
    p.add_argument(
        "--bug", choices=BUGS, default=None, help="break one rule on purpose"
    )
    asyncio.run(ReferenceServer(p.parse_args()).serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
