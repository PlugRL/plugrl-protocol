# plugrl-protocol

[![CI](https://github.com/PlugRL/plugrl-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/PlugRL/plugrl-protocol/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

The wire protocol between a PlugRL **training server**, which holds the
policy and the learning algorithm, and an **env client**, which runs
environments and asks for actions.

The two are separate processes, usually on separate machines, deliberately:
an environment stack and a training stack can then be installed, versioned
and scheduled independently, and an env client need not be Python at all.

**[SPEC.md](SPEC.md) is the specification.** It is normative, it names its
own known defects, and everything in it that can be checked by a test is.

## What is here

| | |
|---|---|
| [`SPEC.md`](SPEC.md) | the protocol, in full |
| [`src/plugrl_protocol/`](src/plugrl_protocol/) | the message types and the msgpack codec — 101 lines |
| [`src/plugrl_protocol/conformance.py`](src/plugrl_protocol/conformance.py) | `plugrl-conformance`, which grades a client clause by clause |
| [`src/plugrl_protocol/server_conformance.py`](src/plugrl_protocol/server_conformance.py) | `plugrl-conformance-server`, which grades a server |
| [`src/plugrl_protocol/reuse.py`](src/plugrl_protocol/reuse.py) | the server's half of the `reuse-feedback-obs` feature |
| [`examples/`](examples/) | two env clients and a reference server written against the spec, sharing no code with PlugRL, and `conformance_server.py`, a shim for `plugrl-conformance` |
| [`tests/`](tests/) | the spec's checkable clauses, as tests |

The codec is small on purpose. It is the piece both sides import, so anything
that could live on one side does; the checker sits beside it rather than in
it, and is not imported by either side at runtime.

## Checking a client

One command, which starts your client once the socket is listening:

```bash
uv run --extra conformance plugrl-conformance \
    --port 8000 --steps 20 --client "./my_client 127.0.0.1 8000 20"
```

It exits non-zero on a violation, so it can sit in a CI job. What it does not
require, because SPEC.md does not, is listed at the top of the module.

That form watches one connection, so it sees the messages and not what the
client did with them. To check that too, have the client run the probe
environment of [SPEC.md section 8.1](SPEC.md#81-the-probe-environment), a
few lines in any language, and add `--probe`:

```bash
uv run --extra conformance plugrl-conformance --probe --scenario all \
    --client "./my_client --probe 127.0.0.1 8000"
```

The checker then works out from each feedback whether the client summed the
chunk's reward, sent the terminal observation, and applied the actions
time-major and in order. It also drives the connection: it speaks late, sends
a metadata frame larger than 1 MiB, closes for a resync, stops the run, and
answers with a text frame, starting the client once per scenario.

## Checking a server

The other direction: a client that drives a training server the way SPEC.md
lets a client behave, and checks what comes back. That covers the action
layout and `env_ids`, ragged batches, large frames, a resync close on each
kind of malformed message with the server staying up afterwards, and the
stop at the end of the run ([SPEC.md section 8.2](SPEC.md#82-checking-a-server)):

```bash
uv run --extra conformance plugrl-conformance-server --port 8000 --state-dim 3 --until-stop
```

[`examples/reference_server.py`](examples/reference_server.py) is a server
written against the specification that trains nothing, and passes.

## The protocol in one screen

Four message types, four lowercase strings. The server speaks first.

```
client                                     server
  |------------ WebSocket handshake ---------->|
  |<---------------- metadata -----------------|
  |------------------ infer ------------------>|   observations
  |<----------------- action ------------------|   an action chunk
  |----------------- feedback ---------------->|   reward, done, next obs
```

Messages are msgpack maps carrying a `message_type`. Arrays travel as

```
{b"__ndarray__": true, b"data": <bin>, b"dtype": "<f4", b"shape": [4, 1, 7]}
```

`dtype` is a numpy typestr — a byte-order character, a kind character and an
item size. It is nearly the only piece of numpy vocabulary on the wire, and
about ten lines of code to parse. The exception is the `text` field; see
[SPEC.md section 3.4](SPEC.md#34-text-is-the-one-real-wart), which does not
pretend otherwise.

Three things a first implementation usually gets wrong, all specified:

- messages **strictly alternate** infer, action, feedback — the server has no
  dispatcher ([section 4.2](SPEC.md#42-the-alternation-invariant));
- the environment set in a `feedback` **need not match** the one in the
  `infer` it follows ([section 4.3](SPEC.md#43-the-env-sets-in-one-cycle-need-not-match));
- the reward in a `feedback` is the **sum over the action chunk**, not the
  last step's ([section 5.4](SPEC.md#54-feedback--client-to-server)).

## Optional features

A server lists the additions to version 1 it implements in its metadata's
`features`, and a client uses one only when it is listed, so an old client
works with a new server and the other way round
([section 10](SPEC.md#10-versioning)).

There is one so far, `reuse-feedback-obs`
([section 10.1](SPEC.md#101-reuse-feedback-obs)). Without it every
observation crosses the link twice: once in the feedback that ends a chunk,
and again in the next infer. With it the infer can say "the one you already
have". `plugrl-conformance --features reuse-feedback-obs` offers it to a
client, and `plugrl-conformance-server` exercises it when the server lists it.

## Relationship to openpi

The serialization is [openpi](https://github.com/Physical-Intelligence/openpi)'s.
`msgpack_numpy.py` is taken from it under Apache-2.0 (attributed in the file
header and in [`NOTICE`](NOTICE)) and reformatted only — **the array encoding
is byte-identical**, so the part of a client that takes real work in C++ or
Rust carries across between the two ecosystems.

The message layer is where they differ: openpi sends a bare observation and
gets a bare action back, which is what serving a policy needs. PlugRL wraps
both in an envelope and adds a `feedback` return channel, which is what
training one needs. [SPEC.md section 9](SPEC.md#9-relationship-to-openpi) has
the full comparison.

## Installing

Not on PyPI. Install from git, pinned to a commit - which is how
`plugrl-server` and `plugrl-env-client` depend on it:

```bash
uv add "plugrl-protocol @ git+https://github.com/PlugRL/plugrl-protocol.git@<commit>"
# or
pip install "plugrl-protocol @ git+https://github.com/PlugRL/plugrl-protocol.git@<commit>"
```

Only Python clients that want the shared codec need this at all. A client in
another language should implement [SPEC.md](SPEC.md) directly, which is what
the C++ example does - and what `plugrl-conformance` will grade it against.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
