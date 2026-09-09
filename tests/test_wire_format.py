"""Contract tests for the wire format.

These values travel between the training server and every env client, which
are separate processes on separate machines running separate dependency
stacks. Changing any of them is a breaking protocol change, and this file
exists to make that break loud rather than silent.
"""

import numpy as np

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.websocket_protocol import (
    SERVER_RESYNC_REASON,
    SERVER_STOP_REASON,
    MessageType,
)


def test_message_type_wire_values_are_stable():
    assert str(MessageType.INFER) == "infer"
    assert str(MessageType.FEEDBACK) == "feedback"
    assert str(MessageType.METADATA) == "metadata"
    assert str(MessageType.ACTION) == "action"


def test_close_reasons_are_stable():
    assert SERVER_STOP_REASON == "plugrl-server-stop"
    assert SERVER_RESYNC_REASON == "plugrl-server-resync"


def test_float_array_survives_round_trip():
    packer = msgpack_numpy.Packer()
    original = np.arange(12, dtype=np.float32).reshape(3, 4)

    restored = msgpack_numpy.unpackb(packer.pack({"obs": original}))["obs"]

    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    np.testing.assert_array_equal(restored, original)


def test_uint8_image_batch_survives_round_trip():
    packer = msgpack_numpy.Packer()
    original = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    original[0, 0, 0] = [255, 128, 0]

    restored = msgpack_numpy.unpackb(packer.pack({"images": original}))["images"]

    assert restored.dtype == np.uint8
    np.testing.assert_array_equal(restored, original)


def test_nested_observation_structure_survives_round_trip():
    """Observations arrive as nested dicts of images, states and a text field."""
    packer = msgpack_numpy.Packer()
    original = {
        "images": {"cam_high": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
        "states": {"joints": np.ones((1, 7), dtype=np.float32)},
        "text": "pick up the red block",
    }

    restored = msgpack_numpy.unpackb(packer.pack(original))

    assert restored["text"] == original["text"]
    np.testing.assert_array_equal(
        restored["images"]["cam_high"], original["images"]["cam_high"]
    )
    np.testing.assert_array_equal(
        restored["states"]["joints"], original["states"]["joints"]
    )
