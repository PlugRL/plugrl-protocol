"""`ObservationCache`, the server's half of SPEC.md section 10.1.

What it must do is give the server, for every infer, the observation a
version 1 client would have sent: the same rows, in `env_indices` order,
however many of them were reused.
"""

from __future__ import annotations

import numpy as np
import pytest

from plugrl_protocol.reuse import ObservationCache, ReuseError


_PER_ROW = object()  # text naming each row's value


def _obs(values, text=_PER_ROW):
    """An observation whose row i is `values[i]`, in states and an image."""
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    if text is _PER_ROW:
        text = np.asarray([f"t{v:g}" for v in values])
    return {
        "images": {"cam": np.broadcast_to(values[:, None, None], (n, 2, 2)).copy()},
        "states": {"obs": values[:, None]},
        "text": text,
    }


def _feedback(cache, envs, values, done=None, text=_PER_ROW):
    done = done or [False] * len(envs)
    cache.on_feedback(
        np.asarray(envs),
        _obs(values, text),
        np.asarray(done),
        np.zeros(len(envs), bool),
    )


def _assert_same(got, want):
    assert set(got) == {"images", "states", "text"}
    for group in ("images", "states"):
        assert set(got[group]) == set(want[group])
        for name in want[group]:
            np.testing.assert_array_equal(got[group][name], want[group][name])
    if isinstance(want["text"], np.ndarray):
        np.testing.assert_array_equal(got["text"], want["text"])
    else:
        assert got["text"] == want["text"]


def test_an_infer_without_reuse_is_passed_through():
    obs = _obs([1, 2])
    assert ObservationCache().complete(np.asarray([0, 1]), obs, None) is obs


def test_reused_rows_come_back_in_env_indices_order():
    cache = ObservationCache()
    _feedback(cache, [0, 1, 2], [10, 11, 12])
    # Env 1 sends a fresh row; envs 2 and 0 reuse theirs.
    got = cache.complete(
        np.asarray([2, 1, 0]), _obs([21]), np.asarray([True, False, True])
    )
    _assert_same(got, _obs([12, 21, 10]))


def test_every_row_reused_sends_an_empty_batch():
    cache = ObservationCache()
    _feedback(cache, [0, 1], [10, 11])
    got = cache.complete(np.asarray([1, 0]), _obs([]), np.asarray([True, True]))
    _assert_same(got, _obs([11, 10]))


def test_the_row_reused_is_the_last_feedback_for_that_env():
    cache = ObservationCache()
    _feedback(cache, [0, 1], [10, 11])
    _feedback(cache, [0], [20])  # only env 0; env 1 keeps its row
    got = cache.complete(np.asarray([0, 1]), _obs([]), np.asarray([True, True]))
    _assert_same(got, _obs([20, 11]))


@pytest.mark.parametrize("ended", ["terminated", "truncated"])
def test_no_reuse_after_a_feedback_that_ended_the_episode(ended):
    cache = ObservationCache()
    cache.on_feedback(
        np.asarray([0, 1]),
        _obs([10, 11]),
        np.asarray([ended == "terminated", False]),
        np.asarray([ended == "truncated", False]),
    )
    with pytest.raises(ReuseError, match="ended its episode"):
        cache.complete(np.asarray([0]), _obs([]), np.asarray([True]))
    # Env 1 did not end, and the next feedback for env 0 makes it reusable.
    cache.complete(np.asarray([1]), _obs([]), np.asarray([True]))
    _feedback(cache, [0], [30])
    got = cache.complete(np.asarray([0]), _obs([]), np.asarray([True]))
    _assert_same(got, _obs([30]))


def test_no_reuse_for_an_env_never_fed_back():
    cache = ObservationCache()
    _feedback(cache, [0], [10])
    with pytest.raises(ReuseError, match="no feedback here"):
        cache.complete(np.asarray([0, 1]), _obs([]), np.asarray([True, True]))


@pytest.mark.parametrize(
    "reuse",
    [
        np.asarray([1, 0]),  # not bool
        np.asarray([True]),  # wrong length
        [True, False],  # not an array
    ],
)
def test_a_malformed_reuse_is_refused(reuse):
    cache = ObservationCache()
    _feedback(cache, [0, 1], [10, 11])
    with pytest.raises(ReuseError, match="bool array"):
        cache.complete(np.asarray([0, 1]), _obs([11]), reuse)


def test_data_batched_to_the_wrong_number_of_rows_is_refused():
    cache = ObservationCache()
    _feedback(cache, [0, 1], [10, 11])
    with pytest.raises(ReuseError, match="leading dimension"):
        # One row reused, so data must have one row, not two.
        cache.complete(np.asarray([0, 1]), _obs([1, 2]), np.asarray([True, False]))


@pytest.mark.parametrize(
    ("feedback_text", "infer_text", "want"),
    [
        (None, None, None),
        ("go", "go", "go"),
        (["a", "b"], ["c"], ["b", "c"]),
        (np.asarray(["a", "b"]), np.asarray(["c"]), np.asarray(["b", "c"])),
        (None, ["c"], ["", "c"]),
    ],
)
def test_text_keeps_the_form_it_was_sent_in(feedback_text, infer_text, want):
    cache = ObservationCache()
    _feedback(cache, [0, 1], [10, 11], text=feedback_text)
    got = cache.complete(
        np.asarray([1, 2]), _obs([12], text=infer_text), np.asarray([True, False])
    )
    if isinstance(want, np.ndarray):
        np.testing.assert_array_equal(got["text"], want)
    else:
        assert got["text"] == want
