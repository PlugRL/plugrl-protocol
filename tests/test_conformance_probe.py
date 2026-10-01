"""The probe checker catches each rule a client can break, and passes one that keeps them.

`examples/raw_client.py --probe` runs the probe environment of SPEC.md
section 8.1 and follows every client clause of section 8. Its `--bug` option
breaks one clause on purpose. Each case below runs the checker against one
bug and expects it to name that clause, so a checker that stopped checking a
clause fails here instead of quietly printing "no violations".
"""

from __future__ import annotations

import pathlib
import socket
import subprocess
import sys

import pytest

pytest.importorskip("websockets")

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLIENT = ROOT / "examples" / "raw_client.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _check(scenario: str, bug: str | None = None) -> subprocess.CompletedProcess:
    port = _free_port()
    client = f'"{sys.executable}" "{CLIENT}" --probe --batch 3 --port {port}'
    if bug:
        client += f" --bug {bug}"
    return subprocess.run(
        [
            sys.executable, "-m", "plugrl_protocol.conformance",
            "--probe", "--scenario", scenario, "--port", str(port),
            "--steps", "8", "--horizon", "4", "--action-dim", "3",
            "--action-dtype", "float64", "--timeout", "40",
            "--client", client,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )  # fmt: skip


def test_a_conforming_client_passes_every_scenario():
    result = _check("all")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no violations" in result.stdout
    for clause in (
        "5.4 rewards is the sum over the steps of the chunk",
        "5.4 a done step reports the observation that step returned",
        "5.3 the action chunk is applied time-major, in order, as sent",
        "7.2 a client reconnects after plugrl-server-resync",
        "7.6 a reconnecting client drops held feedback and starts with infer",
        "7.4 a client treats a text frame as fatal",
        "7.1 a client exits cleanly on plugrl-server-stop",
    ):
        assert f"ok    {clause}" in result.stdout, clause


@pytest.mark.parametrize(
    ("bug", "scenario", "clause"),
    [
        ("early-infer", "basic", "5.1 a client reads metadata before sending anything"),
        ("compress", "basic", "1.1 a client does not offer compression"),
        ("frame-cap", "basic", "1.1 a client accepts frames larger than 1 MiB"),
        ("float32", "basic", "5.3 the action chunk is applied time-major, in order, as sent"),
        ("env-major", "basic", "5.3 the action chunk is applied time-major, in order, as sent"),
        ("last-step-reward", "basic", "5.4 rewards is the sum over the steps of the chunk"),
        ("reset-in-step", "basic", "5.4 a done step reports the observation that step returned"),
        ("resend-feedback", "resync", "7.6 a reconnecting client drops held feedback and starts with infer"),
        ("ignore-stop", "basic", "7.1 a client does not reconnect after plugrl-server-stop"),
        ("ignore-text", "text", "7.4 a client treats a text frame as fatal"),
    ],
)  # fmt: skip
def test_each_broken_rule_is_named(bug, scenario, clause):
    result = _check(scenario, bug)

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"FAIL  {clause}" in result.stdout, result.stdout
