# Put this directory on PYTHONPATH with FV_ORACLE_DUMP_DIR set: every Python
# process of the reference (FastVideo's multiprocessing workers included)
# then installs the oracle dump patches at start (scripts/gpu/upstream/oracle_dump.py).
import os
import sys

if os.environ.get("FV_ORACLE_DUMP_DIR"):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import oracle_dump

        oracle_dump.install()
    except Exception as e:  # noqa: BLE001
        print(f"[oracle_dump] install failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
