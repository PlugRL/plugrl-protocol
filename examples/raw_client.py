"""A PlugRL env client that shares no code with PlugRL.

This exists to test one claim: that the wire protocol can be spoken by a
program written against the specification alone - a robot's onboard controller
in C++, a ROS node, anything that is not this Python codebase.

To make that test mean something, the rules here are strict:

  * nothing from plugrl_protocol, plugrl_env_client or plugrl_server
  * no numpy - arrays are built and parsed with the `array` module, exactly
    as a C++ client would have to
  * only msgpack and websockets, both of which have mature implementations in
    C++, Rust, Go, JavaScript and more

If this completes episodes against a real plugrl-server, the protocol is
portable. If it needs something only Python can express, it is not, and that
is worth knowing before building an argument on it.

Usage:
    python raw_client.py [--host 127.0.0.1] [--port 8000] [--steps 20]

With `--probe` it runs the probe environment of SPEC.md section 8.1 instead
of a fake sensor, executes each action chunk the way section 5.3 says, and
handles the server's close reasons: it reconnects after a resync, dropping the
feedback it was holding, exits 0 when the server stops the run, and treats a
text frame as fatal. That is what `plugrl-conformance --probe` checks:

    python raw_client.py --probe --batch 3 [--host 127.0.0.1] [--port 8000]

`--bug NAME` makes it break one rule on purpose, so the checker's tests can
show that each rule is actually checked.
"""

from __future__ import annotations

import argparse
import sys
import time
from array import array

import msgpack
import websockets.exceptions
import websockets.sync.client

# ---------------------------------------------------------------- wire format
#
# Message types, copied from the protocol - four lowercase strings.
INFER = "infer"
FEEDBACK = "feedback"
METADATA = "metadata"
ACTION = "action"

# An ndarray travels as a map with binary keys:
#
#   {b"__ndarray__": true, b"data": <bin>, b"dtype": "<f4", b"shape": [b, d]}
#
# dtype is a numpy typestr: a byte-order character, a kind character, and an
# item size in bytes. That is the one piece of numpy vocabulary on the wire,
# and it is simple enough to parse by hand - which is what the next few
# functions do, to prove the point.

_ARRAY_CODE = {
    ("f", 4): "f",  # float32
    ("f", 8): "d",  # float64
    ("i", 1): "b",
    ("i", 2): "h",
    ("i", 4): "i",
    ("i", 8): "q",
    ("u", 1): "B",  # uint8, how images arrive
    ("u", 2): "H",
    ("u", 4): "I",
    ("u", 8): "Q",
}

# numpy's bool_ is one byte per element, 0 or 1. Python's array module has no
# boolean typecode, so it is handled as raw bytes - which is what a C++ client
# would do anyway.
_BOOL_KIND = "b"


def _parse_typestr(typestr: str) -> tuple[str, str, int]:
    """'<f4' -> ('<', 'f', 4). '|u1' means byte order is irrelevant."""
    order, kind, size = typestr[0], typestr[1], int(typestr[2:])
    if order == "|":
        order = "<"
    return order, kind, size


def encode_array(values, typestr: str, shape: tuple[int, ...]) -> dict:
    """Pack a flat sequence of numbers into the on-wire ndarray form."""
    order, kind, size = _parse_typestr(typestr)
    if kind == _BOOL_KIND:
        raw = bytes(1 if v else 0 for v in values)
    else:
        buf = array(_ARRAY_CODE[(kind, size)], values)
        if (sys.byteorder == "little") != (order == "<"):
            buf.byteswap()
        raw = buf.tobytes()
    return {
        b"__ndarray__": True,
        b"data": raw,
        b"dtype": typestr,
        b"shape": list(shape),
    }


def decode_array(obj: dict) -> tuple[list, str, list]:
    """Unpack the on-wire ndarray form into (flat values, typestr, shape)."""
    typestr = obj[b"dtype"]
    if isinstance(typestr, bytes):
        typestr = typestr.decode()
    order, kind, size = _parse_typestr(typestr)
    raw = obj[b"data"]
    if kind == _BOOL_KIND:
        values = [b != 0 for b in raw]
    else:
        buf = array(_ARRAY_CODE[(kind, size)])
        buf.frombytes(raw)
        if (sys.byteorder == "little") != (order == "<"):
            buf.byteswap()
        values = list(buf)
    return values, typestr, list(obj[b"shape"])


def looks_like_array(obj) -> bool:
    return isinstance(obj, dict) and b"__ndarray__" in obj


# ---------------------------------------------------------------- fake sensor
#
# A stand-in for whatever the real machine reads. Deterministic, so a failure
# is reproducible.

_seed = 12345


def _next_float() -> float:
    global _seed
    _seed = (1103515245 * _seed + 12345) & 0x7FFFFFFF
    return (_seed % 10000) / 10000.0


def encode_bytes_array(raw: bytes, typestr: str, shape: tuple[int, ...]) -> dict:
    """Wrap already-laid-out bytes, the way a camera driver would hand them over."""
    return {
        b"__ndarray__": True,
        b"data": raw,
        b"dtype": typestr,
        b"shape": list(shape),
    }


def _fake_image(batch: int, h: int, w: int) -> dict:
    # A real camera hands over a buffer; there is no reason to build it element
    # by element. uint8 needs no byte-order handling.
    pattern = bytes(range(256))
    n = batch * h * w * 3
    raw = (pattern * (n // 256 + 1))[:n]
    return encode_bytes_array(raw, "|u1", (batch, h, w, 3))


def make_observation(batch: int, state_dim: int = 10, joint_dim: int = 5) -> dict:
    """The Observation dataclass as it appears on the wire: images, states, text.

    The images are not decoration. dummy-v1 sends a 224x224 and a 112x112 frame
    per step, so including them is what exercises the ~180KB binary payload
    path that a real vision-based policy depends on.
    """
    return {
        "images": {
            "base": _fake_image(batch, 224, 224),
            "wrist": _fake_image(batch, 112, 112),
        },
        "states": {
            "robot_state": encode_array(
                [_next_float() for _ in range(batch * state_dim)],
                "<f8",
                (batch, state_dim),
            ),
            "joint_angles": encode_array(
                [_next_float() for _ in range(batch * joint_dim)],
                "<f8",
                (batch, joint_dim),
            ),
        },
        "text": ["do something"] * batch,
    }


# ---------------------------------------------------------------- the client


def run(host: str, port: int, steps: int, batch: int) -> int:
    uri = f"ws://{host}:{port}"
    print(f"connecting to {uri}")

    packer = msgpack.Packer()

    with websockets.sync.client.connect(
        uri, compression=None, max_size=None, open_timeout=30
    ) as ws:
        # 1. The server speaks first, with metadata.
        meta = msgpack.unpackb(ws.recv(), raw=False, strict_map_key=False)
        if meta.get("message_type") != METADATA:
            print(f"expected {METADATA}, got {meta.get('message_type')!r}")
            return 1
        print(f"metadata: {meta['data']}")

        env_indices = encode_array(range(batch), "<i8", (batch,))
        completed = 0

        for step in range(steps):
            # 2. Ask for an action.
            step_ids = encode_array([step] * batch, "<i8", (batch,))
            ws.send(
                packer.pack(
                    {
                        "message_type": INFER,
                        "data": make_observation(batch),
                        "env_indices": env_indices,
                        "step_ids": step_ids,
                    }
                )
            )

            reply = ws.recv()
            if isinstance(reply, str):
                print(f"server error: {reply}")
                return 1
            action_msg = msgpack.unpackb(reply, raw=False, strict_map_key=False)
            if action_msg.get("message_type") != ACTION:
                print(f"expected {ACTION}, got {action_msg.get('message_type')!r}")
                return 1

            action_field = action_msg["data"]["action"]
            if not looks_like_array(action_field):
                print(f"action arrived as {type(action_field).__name__}, not an array")
                return 1

            values, typestr, shape = decode_array(action_field)
            if 0 in shape:
                print(
                    f"action shape {shape} contains a zero: the server "
                    f"produced no action for this request"
                )
                return 1
            if step == 0:
                print(f"action: dtype={typestr} shape={shape}")
                print(f"        first values: {[round(v, 4) for v in values[:6]]}")

            # 3. Report what happened. The reward is invented; the point is the
            #    shape of the exchange, not the learning signal.
            ws.send(
                packer.pack(
                    {
                        "message_type": FEEDBACK,
                        "env_indices": env_indices,
                        "step_ids": step_ids,
                        "data": {
                            "obs": make_observation(batch),
                            "rewards": encode_array(
                                [_next_float() for _ in range(batch)], "<f4", (batch,)
                            ),
                            "terminated": encode_array([0] * batch, "|b1", (batch,)),
                            "truncated": encode_array([0] * batch, "|b1", (batch,)),
                            "info": {},
                        },
                    }
                )
            )
            completed += 1

        print(f"completed {completed} infer/action/feedback exchanges")
        return 0


# ---------------------------------------------------------------- probe mode
#
# SPEC.md section 8.1. Env i counts the steps of its episode in t, remembers
# the action it applied last in a, pays a reward of 1 per step, and ends its
# episode, terminated, at t = 3 + 2 * (i % 3).

STOP_REASON = "plugrl-server-stop"
RESYNC_REASON = "plugrl-server-resync"

BUGS = (
    "early-infer",  # sends an infer before reading the metadata
    "compress",  # offers permessage-deflate in the handshake
    "frame-cap",  # keeps the websockets library's 1 MiB frame cap
    "float32",  # assumes the action is float32 whatever its typestr says
    "env-major",  # reads the action chunk as [n, H, da]
    "last-step-reward",  # reports the last step's reward, not the chunk's sum
    "reset-in-step",  # resets inside step, so a done step sends the next episode's obs
    "resend-feedback",  # resends the feedback it was holding after a resync
    "ignore-stop",  # reconnects after plugrl-server-stop
    "ignore-text",  # carries on after a text frame instead of stopping
)


class TextFrame(Exception):
    """The server sent text where binary was expected: section 7.4, fatal."""


class ProbeEnv:
    def __init__(self, index: int) -> None:
        self.index = index
        self.length = 3 + 2 * (index % 3)
        self.t = 0
        self.a: list[float] = []

    def step(self, action: list[float]) -> tuple[float, bool]:
        self.t += 1
        self.a = list(action)
        return 1.0, self.t == self.length

    def reset(self) -> None:
        self.t = 0
        self.a = []


def probe_observation(envs: list[ProbeEnv], width: int) -> dict:
    n = len(envs)
    a: list[float] = []
    for env in envs:
        a.extend(env.a if env.a else [0.0] * width)
    return {
        "images": {},
        "states": {
            "t": encode_array([float(env.t) for env in envs], "<f8", (n, 1)),
            "a": encode_array(a, "<f8", (n, width)),
        },
        "text": ["probe"] * n,
    }


def _recv(ws):
    reply = ws.recv()
    if isinstance(reply, str):
        raise TextFrame(reply)
    return msgpack.unpackb(reply, raw=False, strict_map_key=False)


def _probe_session(ws, envs: list[ProbeEnv], bug: str | None, state: dict) -> None:
    """One connection, until the server closes it."""
    packer = msgpack.Packer()
    n = len(envs)
    env_indices = encode_array(range(n), "<i8", (n,))
    step_ids = [0] * n

    def infer() -> bytes:
        return packer.pack(
            {
                "message_type": INFER,
                "data": probe_observation(envs, state["width"]),
                "env_indices": env_indices,
                "step_ids": encode_array(step_ids, "<i8", (n,)),
            }
        )

    if bug == "early-infer":
        ws.send(infer())

    # The server speaks first. Unknown keys - and the probe server sends a
    # large one - are ignored, as section 5.1 says.
    meta = _recv(ws)
    if meta.get("message_type") != METADATA:
        raise RuntimeError(f"expected {METADATA}, got {meta.get('message_type')!r}")
    width = (meta.get("data") or {}).get("action_dim")
    if isinstance(width, int) and width > 0 and not state["stepped"]:
        state["width"] = width

    # A correct client dropped this when the old connection closed.
    if state["held"] is not None:
        ws.send(packer.pack(state["held"]))
        state["held"] = None

    while True:
        ws.send(infer())
        try:
            reply = _recv(ws)
        except TextFrame:
            if bug != "ignore-text":
                raise
            continue  # and ask again, as if nothing had happened
        if reply.get("message_type") != ACTION:
            raise RuntimeError(f"expected {ACTION}, got {reply.get('message_type')!r}")
        field = reply["data"]["action"]
        if bug == "float32":
            field = dict(field)
            field[b"dtype"] = "<f4"
        values, _typestr, shape = decode_array(field)
        horizon, rows = shape[0], shape[1]
        width = 1
        for dim in shape[2:]:
            width *= dim
        state["width"], state["stepped"] = width, True

        # Section 5.3: read env_ids when present, else the request's order.
        env_ids = list(range(n))
        if looks_like_array(reply["data"].get("env_ids")):
            env_ids = [int(v) for v in decode_array(reply["data"]["env_ids"])[0]]

        rewards, terminated = [0.0] * n, [False] * n
        for row, index in enumerate(env_ids[:rows]):
            env = envs[index]
            for k in range(horizon):
                if bug == "env-major":
                    base = (row * horizon + k) * width
                else:  # time-major: [H, n, *da]
                    base = (k * rows + row) * width
                reward, done = env.step(values[base : base + width])
                if bug == "last-step-reward":
                    rewards[index] = reward
                else:
                    rewards[index] += reward
                if done:
                    terminated[index] = True
                    if bug == "reset-in-step":
                        env.reset()
                    break  # a chunk is flushed at the terminal step

        feedback = {
            "message_type": FEEDBACK,
            "env_indices": env_indices,
            "step_ids": encode_array(step_ids, "<i8", (n,)),
            "data": {
                # Before any reset: the observation the terminal step returned.
                "obs": probe_observation(envs, state["width"]),
                "rewards": encode_array(rewards, "<f4", (n,)),
                "terminated": encode_array(terminated, "|b1", (n,)),
                "truncated": encode_array([False] * n, "|b1", (n,)),
                "info": {},
            },
        }
        state["last_feedback"] = feedback
        ws.send(packer.pack(feedback))

        for index, env in enumerate(envs):
            if terminated[index]:
                env.reset()
                step_ids[index] = 0
            else:
                step_ids[index] += 1


def run_probe(host: str, port: int, batch: int, bug: str | None) -> int:
    uri = f"ws://{host}:{port}"
    envs = [ProbeEnv(i) for i in range(batch)]
    state = {"width": 1, "stepped": False, "held": None, "last_feedback": None}
    stops_ignored = 0
    while True:
        try:
            with websockets.sync.client.connect(
                uri,
                compression="deflate" if bug == "compress" else None,
                max_size=2**20 if bug == "frame-cap" else None,
                open_timeout=30,
            ) as ws:
                _probe_session(ws, envs, bug, state)
        except TextFrame as frame:
            print(f"server sent a text frame, which means a server error: {frame}")
            return 2
        except websockets.exceptions.ConnectionClosed as closed:
            reason = closed.rcvd.reason if closed.rcvd is not None else ""
            code = closed.rcvd.code if closed.rcvd is not None else None
            if reason == STOP_REASON:
                if bug == "ignore-stop" and stops_ignored == 0:
                    stops_ignored += 1
                    time.sleep(0.2)
                    continue
                print("the server stopped the run")
                return 0
            if reason == RESYNC_REASON:
                # Section 7.6: the server's state for these envs went with the
                # connection, so the feedback in flight cannot be completed.
                state["held"] = (
                    state["last_feedback"] if bug == "resend-feedback" else None
                )
                print("the server asked for a resync; reconnecting")
                time.sleep(0.2)
                continue
            print(f"connection closed: code {code}, reason {reason!r}")
            return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument(
        "--probe",
        action="store_true",
        help="run the SPEC.md section 8.1 probe env until the server stops the run",
    )
    p.add_argument(
        "--bug", choices=BUGS, default=None, help="break one rule on purpose"
    )
    a = p.parse_args()
    if a.probe:
        return run_probe(a.host, a.port, a.batch, a.bug)
    return run(a.host, a.port, a.steps, a.batch)


if __name__ == "__main__":
    raise SystemExit(main())
