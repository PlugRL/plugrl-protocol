# Reference clients, and a server that grades them

Two env clients written against [the protocol specification](../SPEC.md)
alone, sharing no code with PlugRL. They exist to make one claim checkable
rather than asserted: that an environment can be driven by anything able to
speak WebSocket and msgpack, including a robot's onboard controller.

Both drive a real `plugrl-server` through complete infer/action/feedback
exchanges, and the server advances its training loop as it would for any
other client.

Alongside them, [`conformance_server.py`](conformance_server.py) is a server
that checks a client against SPEC.md clause by clause and says which one it
broke. `plugrl-server` is deliberately forgiving — it validates the
`message_type`, then hands everything to the policy — which is the right
trade for a training run and the wrong one for someone bringing up a client
in a new language.

```bash
python conformance_server.py --port 8000 --steps 20 &
./plugrl_client 127.0.0.1 8000 20
```

It exits non-zero on a violation, so it can sit in a CI job. Both clients
here pass it with one note: they send `text` as a msgpack string array
rather than the `<U` array `plugrl-env-client` sends, which is the
[section 3.4 Gap](../SPEC.md#34-text-is-the-one-real-wart).

The report has two severities, and the distinction is the point. A
**violation** is something `plugrl-server` would reject or mishandle. A
**note** is something it accepts but that differs from what the Python
client does — a portability risk, not a breach. Reporting the second as the
first would mean enforcing rules the protocol does not have.

Options worth knowing: `--horizon`, `--action-dim`, and `--action-dtype`.
The last one is how you find out whether a client really parses the typestr
or has quietly hard-coded float32 — run it with `--action-dtype float64` and
see whether the values it prints are the ones the server sent.

| | `raw_client.py` | `plugrl_client.cpp` |
|---|---|---|
| Language | Python | C++17 |
| Dependencies | `msgpack`, `websockets` | **none** |
| Lines | 275 | 843 |
| Notably absent | numpy, and every `plugrl_*` package | libstdc++ and libc are the only links |

The C++ one is the interesting case. It was written on a machine with no
msgpack library, no WebSocket library and no OpenSSL, so SHA-1, base64,
WebSocket framing and masking, and the msgpack subset the protocol needs are
all in that single file. `ldd` on the result shows `libstdc++`, `libgcc_s`,
`libc` and `libm` and nothing else - which is the situation an embedded
controller is actually in.

```bash
g++ -std=c++17 -O2 -o plugrl_client plugrl_client.cpp
./plugrl_client 127.0.0.1 8000 120        # host port steps
./plugrl_client 127.0.0.1 8000 120 1 224 2  # ... batch img_size cameras
```

```bash
python raw_client.py --host 127.0.0.1 --port 8000 --steps 120
```

## What the protocol asks of a client

1. Connect. The server speaks first, with a `metadata` message.
2. Loop: send `infer`, receive `action`, send `feedback`.

Four message types, four lowercase strings. Observations are nested maps of
images, states and text; arrays travel as

```
{b"__ndarray__": true, b"data": <bin>, b"dtype": "<f4", b"shape": [4, 1, 7]}
```

`dtype` is a numpy typestr - a byte-order character, a kind character and an
item size. It is nearly the only piece of numpy vocabulary on the wire, and
both clients parse it by hand in about ten lines (`_parse_typestr` in the
Python one, `pack_ndarray` and the decode path in the C++ one). Note that
numpy's `bool_` is one byte per element and Python's `array` module has no
boolean typecode, so booleans are handled as raw bytes - which is what a C++
client would do anyway.

Everything else is ordinary: plain WebSocket, standard msgpack, no extension
types. A hand-written HTTP upgrade is enough to get `101 Switching Protocols`
and the first metadata frame.

[`../SPEC.md`](../SPEC.md) is the full statement. Two of its rules these
clients are too simple to exercise, because they drive a fixed set of
environments that never terminates: a feedback's environment set need not
match that of the infer it follows, and the reward is the sum over the
action chunk rather than one step's. `plugrl-env-client`'s
`tests/test_protocol_alternation.py` covers both.

## Caveats

These are demonstrations, not production clients. They do not reconnect, do
not handle the server's resync close reason, and invent their observations
rather than reading a sensor. `plugrl-env-client` is the real one; these show
what the floor looks like.

One difference is worth naming rather than leaving to be discovered. These
clients send the observation's `text` field as a **plain msgpack array of
strings**; `plugrl-env-client` sends it as a numpy `<U` array, which is
fixed-width UTF-32 padded with NULs. Both are accepted today, because the
server forwards observations to the policy without validating them. See
[SPEC.md section 3.4](../SPEC.md#34-text-is-the-one-real-wart) - it is a
protocol defect, not a liberty these examples are taking.
