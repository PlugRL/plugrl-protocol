"""The conformance checker, which now lives in the package.

It moved to `plugrl_protocol.conformance` so that it can be installed and run
as `plugrl-conformance` rather than as a path into somebody's checkout. This
file stays because published records name it: E2 and E7 both quote
`python examples/conformance_server.py ...` as the command they ran, and a
record that points at a file which no longer exists is worth less than a
five-line shim.

Every argument is the same, and `--client COMMAND` is new.
"""

from __future__ import annotations

import pathlib
import sys

# So a fresh checkout works without installing anything first, which is how
# the commands in those records were run.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plugrl_protocol.conformance import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
