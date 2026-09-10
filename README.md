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
| [`examples/`](examples/) | two env clients written against the spec, sharing no code with PlugRL |
| [`tests/`](tests/) | the spec's checkable clauses, as tests |

The package is small on purpose. It is the piece both sides import, so
anything that could live on one side does.

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

```bash
uv add plugrl-protocol      # or: pip install plugrl-protocol
```

Only needed by Python clients that want the shared codec. A client in
another language should implement [SPEC.md](SPEC.md) directly; that is what
the C++ example does.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
