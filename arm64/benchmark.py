"""
Remote benchmark runner for generated ARM64 FFT assembly.

Usage:
    ASM_KEY=... python -m armcrunch.arm64.benchmark
"""
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

from . import fftgen

API_URL = "http://home.hedges.io:1337/run"


@dataclass
class BenchmarkResult:
    """Benchmark and correctness record for one FFT size."""

    n: int
    source_chars: int
    expected_checksum: int
    actual_checksum: int
    benchmark: dict[str, int]


class SandboxApiClient:
    """Thin API wrapper with retry handling for 429/503 responses."""

    def __init__(
        self,
        api_key: str,
        url: str = API_URL,
        max_retries: int = 12,
        request_spacing_seconds: float = 0.25,
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.max_retries = max_retries
        self.request_spacing_seconds = request_spacing_seconds
        self.session = requests.Session()

    @staticmethod
    def _safe_json(response: requests.Response) -> dict[str, Any]:
        """Return decoded JSON body or a fallback dict."""
        try:
            body = response.json()
            if isinstance(body, dict):
                return body
            return {"status": "error", "message": str(body)}
        except ValueError:
            return {"status": "error", "message": response.text}

    @staticmethod
    def _retry_after_seconds(response: requests.Response, attempt: int) -> float:
        """
        Pick retry delay from Retry-After header, JSON retry hints, or backoff.
        """
        header_value = response.headers.get("Retry-After")
        if header_value is not None:
            try:
                return max(0.1, float(header_value)) + random.uniform(0.05, 0.2)
            except ValueError:
                pass

        body = SandboxApiClient._safe_json(response)
        retry_hint = body.get("retry_after_seconds")
        if isinstance(retry_hint, int):
            return max(0.1, float(retry_hint)) + random.uniform(0.05, 0.2)

        return min(10.0, 0.5 * (2**attempt)) + random.uniform(0.05, 0.2)

    def run(
        self,
        *,
        source: str,
        entrypoint: str,
        inputs: dict[str, int] | None = None,
        iterations: int = 1,
        timeout_seconds: int = 10,
    ) -> dict[str, Any]:
        """
        Submit assembly source to the API and return successful JSON response.

        Retries on:
        - 429 Rate limit
        - 503 Execution slot busy
        - transient network/server failures
        """
        if len(source) > fftgen.MAX_SOURCE_CHARS:
            raise ValueError(
                f"Source length {len(source)} exceeds max {fftgen.MAX_SOURCE_CHARS}."
            )

        payload: dict[str, Any] = {
            "source": source,
            "entrypoint": entrypoint,
            "iterations": iterations,
            "timeout_seconds": timeout_seconds,
        }
        if inputs:
            payload["inputs"] = inputs

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: str | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=timeout_seconds + 20,
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                sleep_s = min(10.0, 0.5 * (2**attempt)) + random.uniform(0.05, 0.2)
                time.sleep(sleep_s)
                continue

            if response.status_code in (429, 503):
                delay = self._retry_after_seconds(response, attempt)
                body = self._safe_json(response)
                code = body.get("error_code", "UNKNOWN")
                print(f"Retrying after {delay:.2f}s ({response.status_code}/{code}) ...")
                time.sleep(delay)
                continue

            if response.status_code >= 500:
                last_error = f"server error {response.status_code}"
                sleep_s = min(10.0, 0.5 * (2**attempt)) + random.uniform(0.05, 0.2)
                time.sleep(sleep_s)
                continue

            body = self._safe_json(response)

            if response.status_code == 200:
                if body.get("status") == "ok":
                    time.sleep(self.request_spacing_seconds)
                    return body
                error_code = body.get("error_code", "UNKNOWN")
                message = body.get("message", "runtime error")
                raise RuntimeError(f"Execution failed: {error_code}: {message}")

            error_code = body.get("error_code", f"HTTP_{response.status_code}")
            message = body.get("message", "request failed")
            raise RuntimeError(f"API request failed: {error_code}: {message}")

        raise RuntimeError(f"API retries exhausted ({last_error or 'unknown reason'}).")


def _input_vector(n: int) -> np.ndarray:
    """
    Recreate benchmark harness test data:
      re = (i & 15) - 8
      im = (i & 31) - 16
    """
    i = np.arange(n, dtype=np.int64)
    re = ((i & 15) - 8).astype(np.float64)
    im = ((i & 31) - 16).astype(np.float64)
    return re + 1j * im


def _asm_checksum_from_complex(value: complex, scale: float = fftgen.CHECKSUM_SCALE) -> int:
    """
    Mirror FFT_RETURN_CHECKSUM macro behavior in Python.

    The assembly computes:
      x10 = round(real * scale) as int64
      x11 = round(imag * scale) as int64
      x0  = x10 XOR (x11 << 1)
    with 64-bit wraparound.
    """
    real_i = int(np.rint(value.real * scale))
    imag_i = int(np.rint(value.imag * scale))

    mask = (1 << 64) - 1
    packed = (real_i & mask) ^ (((imag_i & mask) << 1) & mask)

    if packed & (1 << 63):
        packed -= 1 << 64
    return packed


def expected_forward_checksum(n: int) -> int:
    """Compute expected checksum using NumPy FFT output bin 0."""
    vec = _input_vector(n)
    fft_out = np.fft.fft(vec)
    return _asm_checksum_from_complex(fft_out[0])


def _default_iterations(n: int) -> int:
    """Reasonable iteration counts per size under API timeout constraints."""
    return {
        16: 12000,
        64: 8000,
        256: 4000,
        1024: 1500,
    }[n]


def benchmark_size(
    client: SandboxApiClient,
    n: int,
    iterations_override: int | None = None,
    dump_asm_dir: Path | None = None,
) -> BenchmarkResult:
    """Generate source, verify correctness, then benchmark one FFT size."""
    source = fftgen.generate_program(n)
    source_chars = len(source)

    if dump_asm_dir is not None:
        dump_asm_dir.mkdir(parents=True, exist_ok=True)
        asm_path = dump_asm_dir / f"fft_{n}.S"
        asm_path.write_text(source, encoding="utf-8")

    expected = expected_forward_checksum(n)

    correctness = client.run(
        source=source,
        entrypoint=f"_bench_fft_{n}",
        iterations=1,
        timeout_seconds=10,
    )
    actual = int(correctness["output"]["return_value"])

    if actual != expected:
        raise AssertionError(
            f"N={n} checksum mismatch: expected {expected}, got {actual}"
        )

    iterations = iterations_override if iterations_override is not None else _default_iterations(n)
    timed = client.run(
        source=source,
        entrypoint=f"_bench_fft_{n}",
        iterations=iterations,
        timeout_seconds=20,
    )

    bench = timed["benchmark"]
    return BenchmarkResult(
        n=n,
        source_chars=source_chars,
        expected_checksum=expected,
        actual_checksum=actual,
        benchmark=bench,
    )


def _print_summary(results: list[BenchmarkResult]) -> None:
    """Print a compact summary table."""
    print()
    print("FFT Benchmark Summary")
    print(
        "N      src_chars   mean_ns   median_ns   min_ns   max_ns   stddev_ns   "
        "iterations   checksum"
    )
    for result in results:
        bench = result.benchmark
        print(
            f"{result.n:<6} "
            f"{result.source_chars:<10} "
            f"{bench.get('mean_ns', 0):<9} "
            f"{bench.get('median_ns', 0):<11} "
            f"{bench.get('min_ns', 0):<8} "
            f"{bench.get('max_ns', 0):<8} "
            f"{bench.get('stddev_ns', 0):<11} "
            f"{bench.get('iterations', 0):<11} "
            f"{result.actual_checksum}"
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for end-to-end correctness + benchmark runs."""
    parser = argparse.ArgumentParser(description="Benchmark generated ARM64 FFT kernels.")
    parser.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=list(fftgen.SUPPORTED_SIZES),
        choices=fftgen.SUPPORTED_SIZES,
        help="FFT sizes to benchmark.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override benchmark iterations for all sizes.",
    )
    parser.add_argument(
        "--dump-asm-dir",
        type=Path,
        default=None,
        help="Optional directory to save generated .S files.",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("ASM_KEY")
    if not api_key:
        raise SystemExit("ASM_KEY must be set in the environment.")

    client = SandboxApiClient(api_key=api_key)

    results: list[BenchmarkResult] = []
    failures: list[str] = []

    for n in args.sizes:
        print(f"\n--- N={n} ---")
        try:
            result = benchmark_size(
                client=client,
                n=n,
                iterations_override=args.iterations,
                dump_asm_dir=args.dump_asm_dir,
            )
            bench = result.benchmark
            print(
                "OK  "
                f"checksum={result.actual_checksum}  "
                f"mean_ns={bench.get('mean_ns')}  "
                f"median_ns={bench.get('median_ns')}  "
                f"min_ns={bench.get('min_ns')}  "
                f"max_ns={bench.get('max_ns')}"
            )
            results.append(result)
        except Exception as exc:  # broad on purpose for batch mode reporting
            msg = f"N={n} failed: {exc}"
            print(msg)
            failures.append(msg)

    if results:
        _print_summary(results)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()