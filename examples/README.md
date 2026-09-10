# Reference clients

Two env clients written against the protocol specification alone, sharing no
code with PlugRL. They exist to make one claim checkable rather than asserted:
that an environment can be driven by anything able to speak WebSocket and
msgpack, including a robot's onboard controller.

Both drive a real `plugrl-server` through complete infer/action/feedback
exchanges, and the server advances its training loop as it would for any
other client.

| | `raw_client.py` | `plugrl_client.cpp` |
|---|---|---|
| Language | Python | C++17 |
| Dependencies | `msgpack`, `websockets` | **none** |
| Lines | 274 | 633 |
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
item size. It is the only piece of numpy vocabulary on the wire, and both
clients parse it by hand in about ten lines (`_parse_typestr` in the Python
one, `pack_ndarray` and the decode path in the C++ one). Note that numpy's
`bool_` is one byte per element and Python's `array` module has no boolean
typecode, so booleans are handled as raw bytes - which is what a C++ client
would do anyway.

Everything else is ordinary: plain WebSocket, standard msgpack, no extension
types. A hand-written HTTP upgrade is enough to get `101 Switching Protocols`
and the first metadata frame.

## Caveat

These are demonstrations, not production clients. They do not reconnect, do
not handle the server's resync close reason, and invent their observations
rather than reading a sensor. `plugrl-env-client` is the real one; these show
what the floor looks like.
