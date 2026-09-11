# The PlugRL wire protocol

Version 1 · written 2026-09-10 against `plugrl-protocol@22b324c`,
`plugrl-server@06e9f63` and `plugrl-env-client@b9edc5b`.

This document specifies the bytes exchanged between a **training server**,
which holds the policy and the learning algorithm, and an **env client**,
which runs environments and asks for actions. Anything able to speak
WebSocket and msgpack can be an env client: a Python process, a ROS node, a
robot's onboard C++ controller.

The specification is written **from** the implementation, not the other way
round. Where the implementation is under-specified, ambiguous, or where two
of its own servers disagree, this document says so in a box marked **Gap**
rather than inventing a rule. Those boxes are the honest part; treat them as
the protocol's known defect list.

Two reference clients in [`examples/`](examples/) are written against this
document and share no code with PlugRL.

---

## 1. Layers

| Layer | Choice |
|---|---|
| Transport | WebSocket (RFC 6455), unencrypted `ws://` |
| Framing | binary frames only; one message per frame |
| Serialization | msgpack, with two conventions for numeric arrays |
| Schema | four message types, distinguished by a `message_type` string |

There is no version negotiation, no handshake beyond WebSocket's own, and no
extension types. A client that can open a socket and parse msgpack is
seven-eighths of the way there; the remaining eighth is the ndarray
convention in section 3.

### 1.1 Connection options

Both endpoints **MUST** be configured with:

- **`compression = None`.** `permessage-deflate` must not be negotiated.
  Observations are mostly already-compressed image bytes, and deflating them
  costs CPU on the hot path for nothing.
- **`max_size = None`.** No frame size limit. A single observation carrying
  two camera frames is around 180 KB; batched observations and unbounded
  info dicts pass the 1 MiB library default easily.

The server listens on `0.0.0.0:8000` by default.

### 1.2 Authentication

The client **MAY** send an `Authorization: Api-Key <key>` request header
during the WebSocket handshake.

> **Gap — the header is not checked.** `WebSocketEnvClientAgent` sends the
> header when `api_key` is set, and no server in this project reads it. The
> protocol has no authentication. Do not expose a training server to a
> network you do not control.

---

## 2. Message envelope

Every application message is a msgpack **map** whose `"message_type"` key
(a msgpack `str`) holds one of exactly four lowercase values:

| Value | Direction | Reply |
|---|---|---|
| `"metadata"` | server to client | none |
| `"infer"` | client to server | `"action"` |
| `"action"` | server to client | `"feedback"` |
| `"feedback"` | client to server | none |

These four strings are the protocol's only magic constants besides the two
close reasons in section 7. They are pinned by
[`tests/test_wire_format.py`](tests/test_wire_format.py).

An endpoint receiving a message whose `message_type` is not the one it
expects **MUST** treat it as a protocol error (section 7.2). The server does
exactly this; it does not attempt to recover in place.

---

## 3. Serialization

### 3.1 Ordinary values

msgpack maps, arrays, strings, integers, floats, booleans and nil map
directly. Map keys carrying schema names (`"message_type"`, `"data"`,
`"env_indices"`, and so on) are msgpack `str`.

### 3.2 Arrays

A multidimensional numeric array travels as a map with **binary** keys:

```
{
  b"__ndarray__": true,
  b"data":  <bin>,          # raw C-contiguous element bytes
  b"dtype": <str>,          # a numpy typestr, e.g. "<f4"
  b"shape": [d0, d1, ...],  # msgpack array of non-negative integers
}
```

The keys are msgpack **bin**, not **str**. In Python that means unpacking
with `raw=False` yields `bytes` keys; a decoder that only looks for the text
`"__ndarray__"` will not find it. This is inherited from openpi (section 9)
and kept for byte compatibility.

`data` **MUST** be exactly `prod(shape) * itemsize` bytes in C (row-major)
order. There is no stride, offset or Fortran-order field.

A zero-dimensional numpy scalar travels differently:

```
{ b"__npgeneric__": true, b"data": <scalar>, b"dtype": <str> }
```

where `data` is an ordinary msgpack number or boolean. Env clients rarely
produce these; decoders should still handle them, since anything a policy
puts in its response may take this form.

### 3.3 The typestr

`dtype` is a numpy typestr: a **byte-order** character, a **kind**
character, and a decimal **item size in bytes**.

| Byte order | Meaning |
|---|---|
| `<` | little-endian |
| `>` | big-endian |
| `\|` | not applicable (single-byte types) |

| Kind | Meaning | Seen on the wire as |
|---|---|---|
| `b` | boolean | `\|b1` — one byte per element, `0` or `1` |
| `u` | unsigned int | `\|u1` for images; `<u2`, `<u4`, `<u8` |
| `i` | signed int | `<i8` for `env_indices` and `step_ids` |
| `f` | float | `<f4` for rewards and actions, `<f8` for states |
| `U` | unicode | `<U<n>` — see section 3.4 |

Kinds `V` (structured), `O` (object) and `c` (complex) are **rejected at
pack time** and never appear.

A ten-line hand-written parser is enough; both reference clients contain
one. This is the only piece of numpy vocabulary a non-Python client must
learn — with one exception.

### 3.4 Text is the one real wart

`Observation.text` is a numpy array of unicode strings, so it is packed by
the same ndarray path as everything else, and its typestr is `<U<n>` where
`n` is the **character** count of the longest string in the array. Each
element occupies `4n` bytes: **UTF-32 in the stated byte order, NUL-padded
to `n` code points**.

Measured, for `["pick up the red block", "stack"]`:

```
dtype  "<U21"        shape [2]        data 168 bytes
first 8 bytes: 70 00 00 00 69 00 00 00     # 'p', 'i' as UTF-32LE
```

`n` is a property of the message, not of the schema — it changes whenever
the longest prompt changes. A non-Python client that wants to *read* a text
field must therefore implement fixed-width UTF-32 with NUL stripping.

> **Gap — clients disagree about how to write it.**
> `plugrl-env-client` sends `text` as a `<U` array, as above.
> [`examples/raw_client.py`](examples/raw_client.py) sends it as a plain
> msgpack array of strings. Both are accepted, because the server does not
> validate observation contents — it forwards them to the policy, and
> whether the policy copes depends on the policy. Of the policies in
> `plugrl-server`, only `dummy_policy` (which takes `len(obs["text"])`) and
> the openpi transform (which renames `text` to `prompt`) touch the field,
> and both accept either form.
>
> **A future revision should require the msgpack-string-array form** and
> have the server normalise on receipt. Until then, a client that must
> interoperate with the Python one should send the `<U` array.

---

## 4. The exchange

### 4.1 Sequence

```
client                                     server
  |------------ WebSocket handshake ---------->|
  |<---------------- metadata -----------------|   exactly once, server speaks first
  |                                            |
  |------------------ infer ------------------>|   \
  |<----------------- action ------------------|    |  repeats until close
  |                                            |    |
  |   ... zero or more local environment steps ...   |
  |                                            |    |
  |----------------- feedback ---------------->|   /
  |                (no reply)                  |
```

### 4.2 The alternation invariant

**On a given connection, application messages strictly alternate:**

```
metadata  ( infer  action  feedback )*
```

The server's connection handler is a straight-line loop that blocks on
`recv()` for an `infer`, replies with an `action`, then blocks on `recv()`
again for exactly one `feedback`. It has no queue and no dispatcher. A
client that sends two `infer` messages without a `feedback` between them
will have its second `infer` parsed as a `feedback`, fail validation, and be
disconnected with a resync request (section 7.2).

**A `feedback` may arrive many environment steps after the `action` it
answers.** With an action horizon of `H`, the client executes up to `H`
local steps before the chunk is exhausted and feedback is due. The server
waits, up to `FEEDBACK_WAIT_TIMEOUT` = 60 s (section 7.3).

### 4.3 The env sets in one cycle need not match

This is the part an independent implementation is most likely to get wrong.

`infer`, `action` and `feedback` each carry a set of environment indices.
Within one `infer`/`action`/`feedback` cycle:

- the `action`'s `env_ids` **MUST** equal the `infer`'s `env_indices`, in
  the same order — the server asserts this;
- the `feedback`'s `env_indices` **need not equal either of them**, and in a
  multi-environment client routinely does not.

The two sets answer different questions. `infer` carries the environments
whose action chunk has just run out; `feedback` carries the environments
whose chunk finished on this environment step. Once environments
desynchronise — the first time one of them terminates early — those are
different sets.

The pairing between an `action` and the following `feedback` is therefore
**pure flow control, not a semantic association**. The server routes
feedback by env index into per-environment state, not by position in the
stream. An implementer who assumes "the feedback tells me about the actions
I just received" will be wrong as soon as a second environment exists.

The alternation still holds exactly, because an environment can only
re-enter the "needs a chunk" set at the top of the iteration *after* the one
in which it entered the "chunk finished" set. One feedback per action, every
time.

### 4.4 Environment indices are connection-scoped

`env_indices` are small non-negative integers naming environments **within
one connection**. The server keeps its per-environment maps inside the
connection handler, so index `0` from one client and index `0` from another
are unrelated. Clients running several processes against one server do not
need to coordinate index ranges.

---

## 5. Message schemas

Shapes are written with `n` = number of environments in *this* message,
`H` = action horizon, `da` = the environment's action shape (usually a
one-tuple).

### 5.1 `metadata` — server to client

```
{ "message_type": "metadata",
  "data": <map> }
```

Sent once, immediately after the handshake, **before the client sends
anything**. A client MUST read it before its first `infer`.

`data` is a map of descriptive keys. **A client MUST NOT require any of
them**, and MUST ignore keys it does not recognise. What the server puts
there today:

| Key | Meaning |
|---|---|
| `protocol_version` | the version of this document the server implements — `1` |
| `server` | `"plugrl-server"` |
| `server_version` | the package version |
| `algorithm` | the learning algorithm's class name |
| `policy` | the policy's class name |
| `action_horizon` | `H`, the number of steps in an action chunk |
| `action_dim` | the width of one action |

`action_horizon` and `action_dim` are read off the policy, which is free
not to declare them. **A key that is absent means the server does not know,
never that the value is zero or a default** — so a client that needs the
shape and does not find it must be configured with it, exactly as before.
That rule is what makes the rest of the map trustworthy.

The server merges anything its caller passes over the top, so an operator
can add a run identifier or correct a value the introspection got wrong.

> **Historical note.** Through 2026-09-10 this message was always `{}` —
> every entry point defaulted it and nothing filled it in, so the action
> shape had to travel out of band. A client written against that behaviour
> still works, which is why none of these keys are required.

### 5.2 `infer` — client to server

```
{ "message_type": "infer",
  "data":        <observation>,
  "env_indices": <ndarray "<i8" shape [n]>,
  "step_ids":    <ndarray "<i8" shape [n]> }
```

The **observation** is a three-key map, batched along a leading axis of `n`:

```
{ "images": { <name>: <ndarray "|u1" shape [n, h, w, c]>, ... },
  "states": { <name>: <ndarray "<f8" shape [n, d]>,       ... },
  "text":   <see section 3.4, shape [n]> }
```

`images` and `states` may each be empty; camera names and state names are
environment-defined and the server does not check them. Every array under
both keys **MUST** share the same leading dimension `n`.

`n` **MAY** differ between successive `infer` messages on one connection.
This is the normal case: the batch is whichever environments happen to need
a chunk, and it is ragged by construction.

> **Gap — `step_ids` is accepted and ignored.** The client sends, per
> environment, the number of completed action chunks in the current episode,
> reset to `0` when the episode ends. Neither server reads it. It is
> validated for presence only; a client must send it, and may send zeros.

### 5.3 `action` — server to client

```
{ "message_type": "action",
  "data": { "env_ids": <ndarray "<i8" shape [n]>,
            "action":  <ndarray shape [H, n, *da]> } }
```

**The action array is time-major**: horizon first, environment second. The
server produces `[n, H, *da]` internally and transposes on the way out.

The client **MAY** consume any prefix of the horizon and re-plan early; it
**MUST NOT** ask for more steps than `H`. The dtype is the environment's
action dtype and is not renegotiated.

> **Gap — the two servers disagree about this message.**
> `WebSocketAgentServer` sends `env_ids` alongside `action`, as above.
> `RayAgentServer` sends `{"action": ...}` with **no `env_ids` key at all**,
> ignores the request's `env_indices`, keeps one environment's worth of
> state per connection, and buffers the action chunk server-side —
> delivering one step per `infer` when the algorithm sets
> `break_action_chunk`. It is an earlier single-environment dialect that was
> not updated when multi-environment support landed.
>
> **A client conforming to this document should read `env_ids` when present
> and otherwise assume the request's own `env_indices`.** Bringing the Ray
> server to this specification, or retiring it, is tracked work rather than
> a protocol choice.

### 5.4 `feedback` — client to server

```
{ "message_type": "feedback",
  "env_indices": <ndarray "<i8" shape [m]>,
  "step_ids":    <ndarray "<i8" shape [m]>,
  "data": {
    "obs":        <observation, batched to m>,
    "rewards":    <ndarray, float kind, shape [m]>,
    "terminated": <ndarray "|b1" shape [m]>,
    "truncated":  <ndarray "|b1" shape [m]>,
    "info":       <map> } }
```

There is **no reply**. The next thing on the wire is the client's next
`infer`.

`obs` is the observation *after* the last action of the chunk — the
environment's next state, not the one that was sent in `infer`.

**`rewards` is the sum over the chunk, not the last step's reward.** The
client accumulates reward across every environment step of the chunk and
flushes the total when the chunk ends. Credit assignment reaching the
learner is therefore at chunk granularity. A client that reports only the
final step's reward will silently train on a different MDP.

`plugrl-env-client` sends `<f4`. The server does not inspect the width, so a
`<f8` reward is accepted; float32 is the convention rather than a rule, and
`examples/conformance_server.py` reports the difference as a note rather
than a violation.

`terminated` and `truncated` are the flags from the **final** step of the
chunk, carrying Gymnasium's usual distinction: `terminated` means the
episode ended by the environment's own rules, `truncated` means it was cut
short (time limit, external stop). A chunk is also flushed early when either
flag is set, so a terminal transition is never buried inside a chunk.

`info` is a free-form map. When present, its values are expected to be
batched to `m` along a leading axis, and the server slices them per
environment; a value that is not `m`-shaped is passed through to every
environment unchanged. `{}` is valid and is what the reference clients send.

#### Terminal observations

The client **MUST** send the observation that the terminal step returned,
not the first observation of the next episode. This corresponds to
Gymnasium's `AutoresetMode.NEXT_STEP`: the step reporting `done` returns the
terminal observation, and the reset happens on the following call.

An environment that resets inside its own `step()` violates this silently —
nothing raises, and the terminal transition simply carries the wrong
observation into the learner's buffer. `plugrl-env-client` pins the declared
mode of every environment it ships and demonstrates the corruption
executably in its `tests/test_terminal_observation.py`.

---

## 6. Batching on the server

Not part of the wire format, but it determines what a client sees.

The server does not answer an `infer` immediately. It queues the request and
a scheduler drains the queue into one batch, running the policy once for all
of them. A request therefore waits for two things: the scheduler's polling
interval (`SCHEDULER_SLEEP_INTERVAL`, 0.1 ms) and the batch-readiness
condition — by default one queued request per connected client, or
`mini_infer_batch_size` environments when that is set.

Two consequences for a client:

- **Latency is not independent of other clients.** A client alone on a
  server is answered as fast as the scheduler polls. A client sharing one
  with slower peers waits for them, up to `INFER_READY_TIMEOUT` = 5 s, after
  which the server logs a warning and keeps waiting.
- **A client must not assume its request is the whole batch.** Actions come
  back sliced out of a larger tensor; the slice boundaries are the request's
  own `n`.

---

## 7. Closing

### 7.1 Normal shutdown

When the algorithm signals that training is finished, the server closes with
close code **1001 (going away)** and reason:

```
plugrl-server-stop
```

A client seeing this reason should exit rather than reconnect, unless it was
explicitly configured to wait for the next run.

### 7.2 Protocol error — resync

When a message fails validation — wrong `message_type`, missing required
key — the server closes with **1001 (going away)** and reason:

```
plugrl-server-resync
```

This means "your stream and mine are out of step". A client should
reconnect, read the fresh `metadata`, and resume from a new `infer`. Any
feedback it was holding is stale and **MUST** be dropped: the server's
per-environment state went with the closed connection.

### 7.3 Feedback timeout

If 60 s pass after an `action` without a `feedback`, the server closes with
**1001** and reason `Feedback timeout`. Environments slower than that must
be split across more connections, or the constant raised on both sides.

### 7.4 Internal error

An unhandled server-side exception closes with **1011 (internal error)** and
reason `Internal server error.`. Unlike openpi (section 9), PlugRL does
**not** send the traceback as a text frame first.

A client should nonetheless be prepared to receive a **text** frame where it
expected binary, and treat it as a fatal server-side error rather than
attempting to unpack it.

### 7.5 The client stopping

Everything above is the server closing. A client may also stop first - it
has collected the episodes it was asked for, or its operator interrupted it -
and that is not an error. The server keeps no state that outlives the
connection, so a client that disappears costs nothing beyond the feedback it
had not yet sent.

A client that is finished **SHOULD** send a WebSocket close frame with status
**1000 (normal closure)** before dropping the socket, as RFC 6455 section
5.5.1 asks. Nothing breaks without one - the server sees the connection end
either way - but the difference is visible in its logs, as
`ConnectionClosedError: no close frame received or sent` rather than a clean
`ConnectionClosedOK`, and an operator reading those logs should not have to
wonder whether a client crashed.

`examples/conformance_server.py` reports a missing close frame as a note, not
a violation, which is the level this rule deserves.

> **Historical note.** Until 2026-09-11 the C++ reference client did not send
> one. It went unnoticed because in every test until then the *server* ran
> out of steps first and closed the connection itself; E7, where the client
> finishes first, made it visible on all 45 runs.

---

## 8. Conformance checklist

A client conforms to version 1 if it:

- [ ] connects over WebSocket with compression disabled and no frame size cap;
- [ ] reads one `metadata` message before sending anything, and requires no
      particular key in it;
- [ ] emits binary frames containing msgpack maps with a `message_type` str;
- [ ] encodes arrays with **bin** keys `__ndarray__` / `data` / `dtype` /
      `shape`, C-contiguous, and parses the typestr rather than assuming a
      dtype;
- [ ] sends exactly one `feedback` for every `action` received, and never
      two `infer` messages in a row;
- [ ] tolerates its `feedback` env set differing from its `infer` env set;
- [ ] reads `action` as `[H, n, *da]`, time-major, and reads `env_ids` when
      present;
- [ ] reports **chunk-summed** reward, and the **terminal** observation on
      the step that reports done;
- [ ] handles close reasons `plugrl-server-stop` and `plugrl-server-resync`
      differently;
- [ ] treats a text frame as a fatal error.

`examples/conformance_server.py` checks every clause above that is visible
from the server's side of the wire, and reports what it cannot enforce as a
note rather than a failure. Both reference clients pass it with one note:
they send `text` as a msgpack string array rather than a `<U` array, which
is the section 3.4 Gap.

What the harness cannot see is what a client does with the `action` it
receives — reading `env_ids`, honouring the time-major layout, consuming the
horizon in order. Those are checked on the Python side by
`plugrl-env-client`'s `tests/test_protocol_alternation.py`.

---

## 9. Relationship to openpi

PlugRL's serialization is openpi's. `msgpack_numpy.py` is taken from
[openpi](https://github.com/Physical-Intelligence/openpi) (Apache-2.0,
attributed in the file header and in `NOTICE`), reformatted only. **The
array encoding is byte-identical**, deliberately, so that the hard part of
writing a client is shared between the two ecosystems.

The message layer differs:

| | openpi | PlugRL |
|---|---|---|
| First server message | the metadata map, **bare** | `{"message_type": "metadata", "data": ...}` |
| Request | the observation map, bare | envelope + `env_indices` + `step_ids` |
| Response | the action map, bare, plus `server_timing` | envelope + `env_ids` + `action` |
| Return channel | none | `feedback`: reward, done flags, next observation |
| Batching | one observation per request | ragged multi-environment batch |
| Errors | traceback sent as a text frame, then close 1011 | close 1011, no traceback frame |
| Health check | `GET /healthz` | none |

**PlugRL is openpi's serving protocol plus a feedback return channel.** That
is the whole difference, and it is the difference between serving a policy
and training one. An openpi client cannot drive a PlugRL server unmodified,
and vice versa, but the array codec — the part that takes real work in C++
or Rust — carries across unchanged.

---

## 10. Versioning

This is version 1. It has no version field on the wire; section 5.1 notes
that the metadata message is the obvious place to put one, and that it
should be.

Changing any of the four `message_type` strings, the two close reasons, the
ndarray key names, the action array's axis order, or the meaning of
`rewards` is a **breaking** change. `tests/test_wire_format.py` and
`tests/test_spec_conformance.py` exist so that such a change fails a test
rather than a deployment.
