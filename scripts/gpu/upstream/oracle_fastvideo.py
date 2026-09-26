#!/usr/bin/env python3
"""bench_fastvideo.py for the oracle, optionally with dense attention.

``--dense`` (before the bench's own arguments) makes FastVideo's example treat
the run as non-VSA: FLASH_ATTN backend, no compression gate, the same as our
``--dense`` (``H3PipelineOptions::dense``). The oracle uses it as the control
that separates VSA's top-k tile selection (a discrete choice that small
numeric differences can flip) from the dense denoiser math.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    argv = sys.argv[1:]
    dense = "--dense" in argv[: argv.index("--") if "--" in argv else len(argv)]
    if dense:
        argv.remove("--dense")
    src = argv[argv.index("--fastvideo-src") + 1]
    sys.path.insert(0, str(Path(src) / "examples/inference/basic"))
    sys.path.insert(0, str(HERE))
    if dense:
        import basic_fasth3

        basic_fasth3._uses_vsa = lambda args: False
        print("[oracle_fastvideo] dense attention (VSA off)", file=sys.stderr, flush=True)
    import bench_fastvideo

    sys.argv = ["bench_fastvideo.py", *argv]
    return bench_fastvideo.main()


if __name__ == "__main__":
    sys.exit(main())
