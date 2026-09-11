"""Executable clauses of SPEC.md.

`test_wire_format.py` pins the handful of constants a reader can check by
eye. This file pins the claims in SPEC.md that a reader cannot: what the
bytes actually look like on the wire, and which shapes of data the codec
refuses.

Every test here names the section of SPEC.md it enforces. If a test fails,
either the implementation changed and the specification is now a lie, or the
specification was wrong to begin with. Both are worth stopping for.
"""

import msgpack
import numpy as np
import pytest

from plugrl_protocol import msgpack_numpy

NDARRAY_KEYS = {b"__ndarray__", b"data", b"dtype", b"shape"}


def on_the_wire(value):
    """Pack through PlugRL, unpack with plain msgpack.

    Plain msgpack is the whole point: it is what a client with no numpy sees,
    so anything this function returns is something a C++ or Rust client has
    to be able to deal with.
    """
    packed = msgpack_numpy.Packer().pack(value)
    return msgpack.unpackb(packed, raw=False, strict_map_key=False)


class TestArrayEncoding:
    """SPEC section 3.2 - the ndarray convention."""

    def test_keys_are_binary_and_exactly_these_four(self):
        """A decoder matching the *text* "__ndarray__" finds nothing."""
        wire = on_the_wire({"a": np.zeros((2, 3), dtype=np.float32)})["a"]

        assert set(wire) == NDARRAY_KEYS
        assert all(isinstance(k, bytes) for k in wire)
        assert "__ndarray__" not in wire

    def test_data_length_is_exactly_prod_shape_times_itemsize(self):
        wire = on_the_wire({"a": np.zeros((2, 3, 4), dtype=np.float32)})["a"]

        assert list(wire[b"shape"]) == [2, 3, 4]
        assert len(wire[b"data"]) == 2 * 3 * 4 * 4

    def test_bytes_are_c_order_even_for_a_non_contiguous_view(self):
        """A transposed view must ship transposed, not as its base buffer.

        This is the encoding bug a hand-written client is most likely to
        introduce, by reaching for the underlying buffer instead of a
        C-ordered copy.
        """
        original = np.arange(6, dtype=np.uint8).reshape(2, 3).T
        assert not original.flags["C_CONTIGUOUS"]

        wire = on_the_wire({"a": original})["a"]

        assert list(wire[b"shape"]) == [3, 2]
        assert wire[b"data"] == bytes([0, 3, 1, 4, 2, 5])

    def test_big_endian_arrays_declare_their_byte_order(self):
        wire = on_the_wire({"a": np.arange(2, dtype=">f4")})["a"]

        assert wire[b"dtype"] == ">f4"
        assert wire[b"data"][:4] == b"\x00\x00\x00\x00"
        assert wire[b"data"][4:] == b"\x3f\x80\x00\x00"  # 1.0 big-endian

    def test_booleans_are_one_byte_per_element(self):
        """SPEC section 3.3: `|b1`, values 0 or 1 - not a bit field."""
        wire = on_the_wire({"a": np.array([True, False, True])})["a"]

        assert wire[b"dtype"] == "|b1"
        assert wire[b"data"] == b"\x01\x00\x01"

    def test_uint8_images_need_no_byte_order(self):
        wire = on_the_wire({"a": np.zeros((1, 2, 2, 3), dtype=np.uint8)})["a"]

        assert wire[b"dtype"] == "|u1"


class TestScalarEncoding:
    """SPEC section 3.2 - the npgeneric convention."""

    def test_numpy_scalar_uses_a_different_marker_and_carries_a_plain_number(self):
        wire = on_the_wire({"a": np.float32(1.5)})["a"]

        assert set(wire) == {b"__npgeneric__", b"data", b"dtype"}
        assert wire[b"dtype"] == "<f4"
        assert wire[b"data"] == 1.5

    def test_scalar_round_trips_back_to_a_numpy_scalar(self):
        packed = msgpack_numpy.Packer().pack({"a": np.int64(7)})
        restored = msgpack_numpy.unpackb(packed)["a"]

        assert restored == 7
        assert restored.dtype == np.dtype("int64")


class TestRejectedDtypes:
    """SPEC section 3.3 - kinds V, O and c never reach the wire."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(np.zeros(2, dtype=np.complex64), id="complex"),
            pytest.param(np.array([{"a": 1}], dtype=object), id="object"),
            pytest.param(
                np.zeros(2, dtype=[("x", "f4"), ("y", "i4")]), id="structured"
            ),
        ],
    )
    def test_pack_raises_rather_than_shipping_something_unreadable(self, value):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            msgpack_numpy.Packer().pack({"a": value})


class TestTextEncoding:
    """SPEC section 3.4 - the one place numpy leaks past the typestr."""

    def test_text_is_a_fixed_width_utf32_array_not_a_string_list(self):
        wire = on_the_wire({"text": np.asarray(["pick up the red block", "stack"])})[
            "text"
        ]

        assert set(wire) == NDARRAY_KEYS, "text travels as an ndarray, not a list"
        assert wire[b"dtype"] == "<U21", "width is the longest string, in characters"
        assert list(wire[b"shape"]) == [2]
        assert len(wire[b"data"]) == 2 * 21 * 4, "four bytes per code point"
        assert wire[b"data"][:8] == bytes([0x70, 0, 0, 0, 0x69, 0, 0, 0]), "'p', 'i'"

    def test_short_strings_are_nul_padded_to_the_widest(self):
        wire = on_the_wire({"text": np.asarray(["ab", "c"])})["text"]

        assert wire[b"dtype"] == "<U2"
        assert wire[b"data"] == bytes(
            [0x61, 0, 0, 0, 0x62, 0, 0, 0]  # "ab"
            + [0x63, 0, 0, 0, 0, 0, 0, 0]  # "c" + NUL
        )

    def test_the_width_changes_with_the_content(self):
        """Why a client cannot hard-code the text dtype: it is per message."""
        short = on_the_wire({"t": np.asarray(["go"])})["t"][b"dtype"]
        longer = on_the_wire({"t": np.asarray(["go left then right"])})["t"][b"dtype"]

        assert short == "<U2"
        assert longer == "<U18"

    def test_a_plain_string_list_is_also_valid_msgpack(self):
        """SPEC section 3.4 Gap: raw_client.py sends this form instead.

        Nothing in the codec objects. This test documents that the ambiguity
        is real rather than theoretical - the server accepts both, and the
        two shipped clients disagree about which to send.
        """
        wire = on_the_wire({"text": ["pick up the red block", "stack"]})["text"]

        assert wire == ["pick up the red block", "stack"]


class TestObservationTree:
    """SPEC section 5.2 - the observation is a three-key nested map."""

    def test_nested_maps_survive_and_leaves_are_ndarray_maps(self):
        observation = {
            "images": {"cam_high": np.zeros((2, 4, 4, 3), dtype=np.uint8)},
            "states": {"joints": np.ones((2, 7), dtype=np.float64)},
            "text": np.asarray(["a", "b"]),
        }

        wire = on_the_wire(observation)

        assert set(wire) == {"images", "states", "text"}
        assert set(wire["images"]["cam_high"]) == NDARRAY_KEYS
        assert wire["images"]["cam_high"][b"dtype"] == "|u1"
        assert wire["states"]["joints"][b"dtype"] == "<f8"

    def test_an_empty_images_map_is_representable(self):
        """Not every environment has a camera; SPEC section 5.2 allows this."""
        wire = on_the_wire(
            {"images": {}, "states": {"s": np.zeros((1, 3), np.float32)}, "text": ""}
        )

        assert wire["images"] == {}


class TestRoundTrip:
    """What a Python client gets back is what a Python client sent."""

    @pytest.mark.parametrize(
        "array",
        [
            pytest.param(np.arange(12, dtype=np.float32).reshape(3, 4), id="f4"),
            pytest.param(np.arange(3, dtype=np.int64), id="i8"),
            pytest.param(np.array([True, False]), id="b1"),
            pytest.param(np.zeros((1, 2, 2, 3), dtype=np.uint8), id="u1"),
            pytest.param(np.arange(2, dtype=">f4"), id="big-endian"),
            pytest.param(np.arange(6, dtype=np.uint8).reshape(2, 3).T, id="transposed"),
            pytest.param(np.asarray(["pick up", "the block"]), id="text"),
            pytest.param(np.zeros((0, 4), dtype=np.float32), id="empty"),
        ],
    )
    def test_dtype_shape_and_values_are_preserved(self, array):
        packed = msgpack_numpy.Packer().pack({"a": array})
        restored = msgpack_numpy.unpackb(packed)["a"]

        assert restored.dtype == array.dtype
        assert restored.shape == array.shape
        np.testing.assert_array_equal(restored, array)
