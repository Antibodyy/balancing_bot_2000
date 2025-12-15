from __future__ import annotations

import sys


def pytest_configure(config):
    """Strip pytest CLI arguments when running mpc_vs_lqr-only commands.

    Several legacy script-style tests parse argparse options at import time and
    choke on pytest's `-k` / `--disable-warnings` flags. When the user requests
    the MPC-vs-LQR comparison specifically, scrub sys.argv so those scripts see
    an empty argument list and use defaults, avoiding premature SystemExit.
    """
    keyword = getattr(config.option, "keyword", "")
    if keyword and "mpc_vs_lqr" in keyword:
        sys.argv = [sys.argv[0]]

