"""The server checker passes a conforming server and names each rule a server breaks.

`examples/reference_server.py` implements every server clause of SPEC.md and
trains nothing. Its `--bug` option breaks one clause on purpose; each case
below runs `plugrl-conformance-server` against one bug and expects it to
name that clause.
"""

from __future__ import annotations

import pathlib
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("websockets")

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = ROOT / "examples" / "reference_server.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _check(bug: str | None = None, *extra: str) -> subprocess.CompletedProcess:
    port = _free_port()
    command = [sys.executable, str(SERVER), "--port", str(port), "--steps", "60"]
    if bug:
        command += ["--bug", bug]
    server = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        for _ in range(100):  # until it listens
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        return subprocess.run(
            [
                sys.executable, "-m", "plugrl_protocol.server_conformance",
                "--port", str(port), "--timeout", "5", *extra,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )  # fmt: skip
    finally:
        server.kill()
        server.wait()


def test_a_conforming_server_passes_every_phase():
    result = _check(None, "--until-stop")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no violations" in result.stdout
    for clause in (
        "5.1 the server speaks first, with metadata",
        "5.3 action is time-major, [H, n, *da]",
        "4.3 action env_ids equal the infer's env_indices, in order",
        "4.3 the server accepts a feedback env set unlike the infer's",
        "1.1 the server accepts a frame larger than 1 MiB",
        "7.2 a malformed infer closes the connection for a resync",
        "4.2 two infers in a row close the connection for a resync",
        "5.4 an info that cannot be split per env closes the connection for a resync",
        "7.2 the server survives a client's protocol error",
        "4.4 env indices are connection-scoped: two clients can both use 0",
        "7.1 the server ends a run with plugrl-server-stop",
        "10.1 the server offers reuse-feedback-obs",
        "10.1 the server answers an infer that reuses observations",
        "10.1 the server answers an infer with every row reused",
        "10.1 a reuse the server cannot honour closes for a resync",
    ):
        assert f"ok    {clause}" in result.stdout, clause


@pytest.mark.parametrize(
    ("bug", "extra", "clause"),
    [
        ("no-metadata", (), "5.1 the server speaks first, with metadata"),
        ("env-major", (), "5.3 action is time-major, [H, n, *da]"),
        ("no-env-ids", (), "4.3 action env_ids equal the infer's env_indices, in order"),
        ("wrong-horizon", (), "5.1 the action matches the metadata's action_horizon"),
        ("lenient", (), "7.2 a malformed infer closes the connection for a resync"),
        ("crash-on-info", (), "7.2 the server survives a client's protocol error"),
        ("plain-stop", ("--until-stop",), "7.1 the server ends a run with plugrl-server-stop"),
        ("lenient-reuse", (), "10.1 a reuse the server cannot honour closes for a resync"),
    ],
)  # fmt: skip
def test_each_broken_rule_is_named(bug, extra, clause):
    result = _check(bug, *extra)

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"FAIL  {clause}" in result.stdout, result.stdout
