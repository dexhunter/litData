"""Baseline timing for ``litdata.streaming.serializers.NumpySerializer.deserialize``.

Workload: 20_000 uint8 (32, 32, 3) image arrays serialized with ``NumpySerializer``,
the per-sample decode that ``StreamingDataset`` runs for every multi-dimensional
numpy leaf. The timed region is one decode pass over all 20_000 blobs.
End-to-end step: one full epoch of ``StreamingDataset`` over an optimized
20_000-sample dataset (index, image, label, text); the script prints the
decode share of that epoch.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from litdata import StreamingDataset, optimize
from litdata.streaming.serializers import NumpySerializer

N = 20_000
REPEATS = 9


def make(i: int) -> dict:
    rng = np.random.default_rng(i)
    return {"index": i, "image": rng.integers(0, 255, (32, 32, 3), dtype=np.uint8), "label": i % 10, "text": f"sample {i}"}


def main() -> None:
    serializer = NumpySerializer()
    rng = np.random.default_rng(0)
    arrays = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(N)]
    blobs = [serializer.serialize(a)[0] for a in arrays]
    for a, b in zip(arrays[:100], blobs[:100]):
        out = serializer.deserialize(b)
        if out.shape != a.shape or out.dtype != a.dtype or not np.array_equal(out, a) or not out.flags.writeable:
            raise SystemExit("decode did not round-trip")

    def run() -> None:
        for b in blobs:
            serializer.deserialize(b)

    out_dir = os.path.join(tempfile.mkdtemp(prefix="osspr-litdata-"), "data")
    optimize(fn=make, inputs=list(range(N)), output_dir=out_dir, chunk_bytes="16MB", num_workers=1)

    def epoch() -> int:
        return sum(1 for _ in StreamingDataset(out_dir))

    run(); epoch()
    samples_ms: list[float] = []
    epoch_ms: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter(); run(); samples_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter(); epoch(); epoch_ms.append((time.perf_counter() - started) * 1000.0)

    median = statistics.median(samples_ms)
    noise = (max(samples_ms) - min(samples_ms)) / median * 100.0
    e2e = statistics.median(epoch_ms)
    print(f"baseline_median_ms={median:.4f}")
    print(f"baseline_noise_pct={noise:.4f}")
    print("runs_ms=" + ",".join(f"{s:.4f}" for s in samples_ms))
    print(f"e2e_epoch_median_ms={e2e:.4f}")
    print(f"share_of_e2e_pct={median / e2e * 100.0:.2f}")


if __name__ == "__main__":
    main()
