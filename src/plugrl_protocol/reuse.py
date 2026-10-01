"""The `reuse-feedback-obs` feature of SPEC.md section 10.1, for servers.

An `infer` that marks a row in `reuse` sends no observation for that env: the
server takes it from the env's last `feedback` on the same connection. A
server keeps one `ObservationCache` per connection, feeds it every feedback,
and asks it for the full observation of every infer:

    cache = ObservationCache()
    ...
    observation = cache.complete(infer["env_indices"], infer["data"], infer.get("reuse"))
    ...
    cache.on_feedback(env_indices, data["obs"], data["terminated"], data["truncated"])

`complete` raises `ReuseError` for a reuse SPEC.md does not allow, and the
server then closes the connection for a resync.
"""

from __future__ import annotations

import numpy as np

REUSE_FEEDBACK_OBS = "reuse-feedback-obs"


class ReuseError(ValueError):
    """An infer reuses an observation the server does not hold."""


def _rows(observation, n: int, what: str) -> list[dict]:
    """Split an observation batched to `n` into `n` observations of one row."""
    if not isinstance(observation, dict) or set(observation) != {
        "images",
        "states",
        "text",
    }:
        raise ReuseError(f"{what} is not a map of images, states and text")
    rows = [{"images": {}, "states": {}, "text": None} for _ in range(n)]
    for group in ("images", "states"):
        arrays = observation[group]
        if not isinstance(arrays, dict):
            raise ReuseError(f"{what}: {group} is not a map")
        for name, array in arrays.items():
            if not (
                isinstance(array, np.ndarray)
                and array.ndim >= 1
                and array.shape[0] == n
            ):
                raise ReuseError(
                    f"{what}: {group}[{name!r}] has leading dimension "
                    f"{getattr(array, 'shape', (None,))[0]}, expected {n}"
                )
            for i in range(n):
                rows[i][group][name] = array[i : i + 1]
    text = observation["text"]
    for i in range(n):
        if isinstance(text, np.ndarray) and text.ndim >= 1:
            rows[i]["text"] = text[i : i + 1]
        elif isinstance(text, (list, tuple)):
            rows[i]["text"] = [text[i]] if i < len(text) else []
        else:
            rows[i]["text"] = text
    return rows


def _merge(rows: list[dict], template) -> dict:
    """Batch observations of one row back together, in order."""
    if not rows:
        return template
    out: dict = {"images": {}, "states": {}}
    for group in ("images", "states"):
        names = list(rows[0][group])
        for row in rows[1:]:
            if list(row[group]) != names:
                raise ReuseError(
                    f"the rows' {group} keys differ: {names} and {list(row[group])}"
                )
        for name in names:
            out[group][name] = np.concatenate(
                [row[group][name] for row in rows], axis=0
            )
    texts = [row["text"] for row in rows]
    if all(isinstance(t, np.ndarray) for t in texts):
        out["text"] = np.concatenate(texts, axis=0)
    elif all(isinstance(t, list) for t in texts):
        out["text"] = [x for t in texts for x in t]
    elif all(t is None for t in texts):
        out["text"] = None
    elif all(isinstance(t, str) for t in texts) and len(set(texts)) == 1:
        out["text"] = texts[0]
    else:
        # Rows that disagree on how text is sent: one string per row.
        flat: list = []
        for t in texts:
            if isinstance(t, np.ndarray):
                flat.extend(t.tolist())
            elif isinstance(t, list):
                flat.extend(t)
            else:
                flat.append(t)
        out["text"] = ["" if x is None else str(x) for x in flat]
    return out


class ObservationCache:
    """One connection's last feedback observation per env, where it may be reused."""

    def __init__(self) -> None:
        # None: the env's last feedback ended its episode, so its observation
        # is the terminal one, not where the next chunk starts.
        self._rows: dict[int, dict | None] = {}

    def on_feedback(self, env_indices, observation, terminated, truncated) -> None:
        envs = [int(e) for e in np.asarray(env_indices).tolist()]
        rows = _rows(observation, len(envs), "the feedback observation")
        done = np.logical_or(np.asarray(terminated), np.asarray(truncated)).tolist()
        for env, row, ended in zip(envs, rows, done):
            self._rows[env] = None if ended else row

    def complete(self, env_indices, observation, reuse) -> dict:
        """The infer's observation for every env in `env_indices`."""
        if reuse is None:
            return observation
        envs = [int(e) for e in np.asarray(env_indices).tolist()]
        if not (
            isinstance(reuse, np.ndarray)
            and reuse.dtype == np.bool_
            and reuse.shape == (len(envs),)
        ):
            raise ReuseError(f"reuse is not a bool array of length {len(envs)}")
        sent = _rows(observation, int((~reuse).sum()), "the infer observation")
        rows, it = [], iter(sent)
        for env, reused in zip(envs, reuse.tolist()):
            if not reused:
                rows.append(next(it))
                continue
            if env not in self._rows:
                raise ReuseError(
                    f"env {env} reuses an observation, but it has had no feedback here"
                )
            if self._rows[env] is None:
                raise ReuseError(
                    f"env {env} reuses an observation, but its last feedback ended its "
                    "episode; the reset observation was never sent"
                )
            rows.append(self._rows[env])
        return _merge(rows, observation)
