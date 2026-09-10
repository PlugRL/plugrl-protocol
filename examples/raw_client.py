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
"""

from __future__ import annotations

import argparse
import sys
from array import array

import msgpack
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch", type=int, default=1)
    a = p.parse_args()
    return run(a.host, a.port, a.steps, a.batch)


if __name__ == "__main__":
    raise SystemExit(main())
