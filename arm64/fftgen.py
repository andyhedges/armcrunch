"""
ARM64 NEON radix-4 FFT assembly generator.

This module emits compact ARM64 assembly source that fits the sandbox
character source limit while still providing:
- Forward FFT (radix-4 DIT)
- Inverse FFT (conjugated twiddles + 1/N scaling)
- Pointwise complex multiply
- Benchmark harness entrypoints

Generated kernels operate on interleaved complex doubles in memory (AoS)
and use ldp/stp for scalar complex pair load/store, and ld2/st2 for
vectorized (.2d) operations in the pointwise multiply kernel.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_SIZES: tuple[int, ...] = (16, 64, 256, 1024)
MAX_SOURCE_CHARS = 524288
CHECKSUM_SCALE = 1_000_000.0


@dataclass(frozen=True)
class StagePlan:
    """Metadata for one radix-4 FFT stage."""

    stage: int
    m: int
    quarter: int
    groups: int
    stride_bytes: int
    group_span_bytes: int


def _is_power_of_four(n: int) -> bool:
    """Return True when n is an exact power of four."""
    if n <= 0:
        return False
    while n % 4 == 0:
        n //= 4
    return n == 1


def _validate_supported_size(n: int) -> None:
    """Validate FFT length constraints supported by this generator."""
    if n not in SUPPORTED_SIZES:
        raise ValueError(f"Unsupported FFT size {n}. Supported sizes: {SUPPORTED_SIZES}.")
    if not _is_power_of_four(n):
        raise ValueError(f"FFT size {n} must be a power of four.")


def _radix4_stage_count(n: int) -> int:
    """Compute number of radix-4 stages for n."""
    stages = 0
    value = n
    while value > 1:
        if value % 4 != 0:
            raise ValueError(f"{n} is not compatible with pure radix-4 staging.")
        value //= 4
        stages += 1
    return stages


def _stage_plan(n: int) -> list[StagePlan]:
    """Build stage plan used by forward/inverse FFT emitters."""
    stage_count = _radix4_stage_count(n)
    plans: list[StagePlan] = []
    quarter = 1
    for stage in range(stage_count):
        m = quarter * 4
        plans.append(
            StagePlan(
                stage=stage,
                m=m,
                quarter=quarter,
                groups=n // m,
                stride_bytes=quarter * 16,
                group_span_bytes=m * 16,
            )
        )
        quarter *= 4
    return plans


def _format_f64(value: float) -> str:
    """Format float for assembly `.double` with deterministic compact text."""
    if abs(value) < 5e-17:
        value = 0.0
    return f"{value:.17g}"


def _mov_imm(reg: str, value: int) -> list[str]:
    """
    Emit movz/movk sequence for a 64-bit immediate.

    This intentionally avoids pseudo-`mov` for large immediates.
    """
    value_u = value & ((1 << 64) - 1)
    chunks = [(value_u >> shift) & 0xFFFF for shift in (0, 16, 32, 48)]

    if all(chunk == 0 for chunk in chunks):
        return [f"movz {reg}, #0"]

    lines: list[str] = []
    first = True
    for idx, chunk in enumerate(chunks):
        if chunk == 0 and first:
            continue
        shift = idx * 16
        if first:
            if shift == 0:
                lines.append(f"movz {reg}, #{chunk}")
            else:
                lines.append(f"movz {reg}, #{chunk}, lsl #{shift}")
            first = False
        elif chunk != 0:
            lines.append(f"movk {reg}, #{chunk}, lsl #{shift}")
    return lines


def _emit_label_address(reg: str, label: str) -> list[str]:
    """Emit adrp/add pair for a data label address."""
    return [f"adrp {reg}, {label}@PAGE", f"add {reg}, {reg}, {label}@PAGEOFF"]


def generate_twiddles(n: int) -> str:
    """
    Generate forward twiddle tables for an n-point radix-4 FFT.

    Twiddles are stage-major and interleaved complex doubles:
    for each stage and each local index j, emit W1/W2/W3 triples.
    """
    _validate_supported_size(n)
    lines: list[str] = [
        ".data",
        ".p2align 7",
        "_const_scale_1e6:",
        f"    .double {_format_f64(CHECKSUM_SCALE)}",
        f"_const_inv_n_{n}:",
        f"    .double {_format_f64(1.0 / n)}",
    ]

    for plan in _stage_plan(n):
        if plan.stage == 0:
            continue
        lines.append(".p2align 7")
        lines.append(f"_tw_fwd_{n}_s{plan.stage}:")
        for j in range(plan.quarter):
            for tw_mult in (1, 2, 3):
                angle = -2.0 * math.pi * tw_mult * j / plan.m
                re = math.cos(angle)
                im = math.sin(angle)
                lines.append(f"    .double {_format_f64(re)}, {_format_f64(im)}")

    return "\n".join(lines)


def generate_fft_macros() -> str:
    """
    Compatibility shim kept for callers; macro directives were removed.

    All previously macro-expanded instruction sequences are now emitted inline
    by Python helper functions.
    """
    return ""


def _emit_cmul_d(ar: str, ai: str, wr: str, wi: str, out_r: str, out_i: str) -> list[str]:
    """
    Emit scalar complex multiply: (ar + i*ai) * (wr + i*wi).

    Uses fmadd/fmsub (4-operand scalar FMA) since scalar fmla/fmls
    (3-operand tied accumulator) does not exist on AArch64.

    fmsub Dd, Dn, Dm, Da  =>  Da - Dn*Dm
    fmadd Dd, Dn, Dm, Da  =>  Da + Dn*Dm
    """
    return [
        f"fmul {out_r}, {ar}, {wr}",          # out_r = ar * wr
        f"fmsub {out_r}, {ai}, {wi}, {out_r}", # out_r = out_r - ai * wi  (= ar*wr - ai*wi)
        f"fmul {out_i}, {ar}, {wi}",           # out_i = ar * wi
        f"fmadd {out_i}, {ai}, {wr}, {out_i}", # out_i = out_i + ai * wr  (= ar*wi + ai*wr)
    ]


def _emit_cmul_v2(ar: str, ai: str, br: str, bi: str, out_r: str, out_i: str) -> list[str]:
    """
    Emit vector complex multiply for two packed complex doubles (.2d).

    Vector fmla/fmls (.2d tied 3-operand form) IS valid on AArch64 NEON.
    """
    return [
        f"fmul {out_r}.2d, {ar}.2d, {br}.2d",
        f"fmls {out_r}.2d, {ai}.2d, {bi}.2d",
        f"fmul {out_i}.2d, {ar}.2d, {bi}.2d",
        f"fmla {out_i}.2d, {ai}.2d, {br}.2d",
    ]


def _emit_load_twiddles(twptr: str) -> list[str]:
    """
    Emit loads of W1/W2/W3 twiddle pairs with post-increment.

    Uses ldp (scalar double pair) instead of ld2 which requires .2d minimum.
    """
    return [
        f"ldp d16, d17, [{twptr}], #16",
        f"ldp d18, d19, [{twptr}], #16",
        f"ldp d20, d21, [{twptr}], #16",
    ]


def _emit_conj_twiddles() -> list[str]:
    """Emit in-register conjugation of loaded twiddles."""
    return [
        "fneg d17, d17",
        "fneg d19, d19",
        "fneg d21, d21",
    ]


def _emit_r4_bfly_notwiddle() -> list[str]:
    """Emit radix-4 butterfly without twiddle multiplies."""
    return [
        "fadd d8, d0, d4",
        "fadd d9, d1, d5",
        "fsub d10, d0, d4",
        "fsub d11, d1, d5",
        "fadd d12, d2, d6",
        "fadd d13, d3, d7",
        "fsub d14, d2, d6",
        "fsub d15, d3, d7",
        # multiply (d14 + i*d15) by -i: result = (d15, -d14)
        "fmov d30, d14",
        "fmov d14, d15",
        "fneg d15, d30",
        # combine
        "fadd d0, d8, d12",
        "fadd d1, d9, d13",
        "fadd d2, d10, d14",
        "fadd d3, d11, d15",
        "fsub d4, d8, d12",
        "fsub d5, d9, d13",
        "fsub d6, d10, d14",
        "fsub d7, d11, d15",
    ]


def _emit_r4_bfly_3twiddle() -> list[str]:
    """Emit radix-4 butterfly with twiddle multiplies for lanes 1..3."""
    lines: list[str] = []
    lines.extend(_emit_cmul_d("d2", "d3", "d16", "d17", "d22", "d23"))
    lines.extend(_emit_cmul_d("d4", "d5", "d18", "d19", "d24", "d25"))
    lines.extend(_emit_cmul_d("d6", "d7", "d20", "d21", "d26", "d27"))
    lines.extend(
        [
            "fmov d2, d22",
            "fmov d3, d23",
            "fmov d4, d24",
            "fmov d5, d25",
            "fmov d6, d26",
            "fmov d7, d27",
        ]
    )
    lines.extend(_emit_r4_bfly_notwiddle())
    return lines


def _emit_return_checksum(ptr_reg: str) -> list[str]:
    """Emit checksum return sequence from output[0]."""
    lines: list[str] = [f"ldp d0, d1, [{ptr_reg}]"]
    lines.extend(_emit_label_address("x9", "_const_scale_1e6"))
    lines.extend(
        [
            "ldr d31, [x9]",
            "fmul d0, d0, d31",
            "fmul d1, d1, d31",
            "frintn d0, d0",
            "frintn d1, d1",
            "fcvtzs x10, d0",
            "fcvtzs x11, d1",
            "lsl x11, x11, #1",
            "eor x0, x10, x11",
        ]
    )
    return lines


def _emit_stage_loop(n: int, plan: StagePlan, symbol_prefix: str, inverse: bool) -> list[str]:
    """Emit one nested-loop radix-4 stage block."""
    group_label = f"L_{symbol_prefix}_s{plan.stage}_group"
    inner_label = f"L_{symbol_prefix}_s{plan.stage}_inner"

    lines: list[str] = [
        f"    // stage {plan.stage}: m={plan.m}, quarter={plan.quarter}, groups={plan.groups}",
    ]
    lines.extend(f"    {inst}" for inst in _mov_imm("x9", plan.groups))
    lines.append("    mov x10, x8")

    if plan.stage > 0:
        lines.extend(f"    {inst}" for inst in _emit_label_address("x17", f"_tw_fwd_{n}_s{plan.stage}"))

    lines.append(f"{group_label}:")
    lines.append("    mov x12, x10")
    lines.append(f"    add x13, x12, #{plan.stride_bytes}")
    lines.append(f"    add x14, x13, #{plan.stride_bytes}")
    lines.append(f"    add x15, x14, #{plan.stride_bytes}")
    if plan.stage > 0:
        lines.append("    mov x11, x17")

    lines.extend(f"    {inst}" for inst in _mov_imm("x16", plan.quarter))
    lines.append(f"{inner_label}:")
    lines.append("    prfm pldl1keep, [x12, #256]")
    lines.append("    prfm pldl1keep, [x13, #256]")
    lines.append("    prfm pldl1keep, [x14, #256]")
    lines.append("    prfm pldl1keep, [x15, #256]")
    # Load 4 complex values as scalar double pairs
    lines.append("    ldp d0, d1, [x12]")
    lines.append("    ldp d2, d3, [x13]")
    lines.append("    ldp d4, d5, [x14]")
    lines.append("    ldp d6, d7, [x15]")

    if plan.stage == 0:
        lines.extend(f"    {inst}" for inst in _emit_r4_bfly_notwiddle())
    else:
        lines.append("    prfm pldl1keep, [x11, #256]")
        lines.extend(f"    {inst}" for inst in _emit_load_twiddles("x11"))
        if inverse:
            lines.extend(f"    {inst}" for inst in _emit_conj_twiddles())
        lines.extend(f"    {inst}" for inst in _emit_r4_bfly_3twiddle())

    # Store 4 complex values with post-increment
    lines.append("    stp d0, d1, [x12], #16")
    lines.append("    stp d2, d3, [x13], #16")
    lines.append("    stp d4, d5, [x14], #16")
    lines.append("    stp d6, d7, [x15], #16")
    lines.append("    subs x16, x16, #1")
    lines.append(f"    b.ne {inner_label}")
    lines.append(f"    add x10, x10, #{plan.group_span_bytes}")
    lines.append("    subs x9, x9, #1")
    lines.append(f"    b.ne {group_label}")

    return lines


def generate_fft_forward(n: int) -> str:
    """
    Generate complete forward radix-4 FFT function for n complex points.

    Calling convention:
    - x0: pointer to AoS complex doubles (in-place)
    - x0 return: checksum derived from output[0]
    """
    _validate_supported_size(n)
    func = f"_fft_fwd_{n}"
    symbol_prefix = f"fwd_{n}"

    lines: list[str] = [
        ".text",
        ".p2align 4",
        f".globl {func}",
        f"{func}:",
        "    mov x8, x0",
    ]

    for plan in _stage_plan(n):
        lines.extend(_emit_stage_loop(n=n, plan=plan, symbol_prefix=symbol_prefix, inverse=False))

    lines.extend(f"    {inst}" for inst in _emit_return_checksum("x8"))
    lines.append("    ret")
    return "\n".join(lines)


def generate_fft_inverse(n: int) -> str:
    """
    Generate complete inverse FFT function for n complex points.

    This kernel reuses forward twiddles by conjugating them in-register and
    applies 1/N scaling after all stages.
    """
    _validate_supported_size(n)
    func = f"_fft_inv_{n}"
    symbol_prefix = f"inv_{n}"
    scale_label = f"L_inv_{n}_scale"

    lines: list[str] = [
        ".text",
        ".p2align 4",
        f".globl {func}",
        f"{func}:",
        "    mov x8, x0",
    ]

    for plan in _stage_plan(n):
        lines.extend(_emit_stage_loop(n=n, plan=plan, symbol_prefix=symbol_prefix, inverse=True))

    lines.append("    // inverse scaling by 1/N")
    lines.extend(f"    {inst}" for inst in _emit_label_address("x9", f"_const_inv_n_{n}"))
    lines.append("    ldr d31, [x9]")
    lines.append("    mov x10, x8")
    lines.extend(f"    {inst}" for inst in _mov_imm("x11", n))
    lines.append(f"{scale_label}:")
    lines.append("    ldp d0, d1, [x10]")
    lines.append("    fmul d0, d0, d31")
    lines.append("    fmul d1, d1, d31")
    lines.append("    stp d0, d1, [x10], #16")
    lines.append("    subs x11, x11, #1")
    lines.append(f"    b.ne {scale_label}")

    lines.extend(f"    {inst}" for inst in _emit_return_checksum("x8"))
    lines.append("    ret")
    return "\n".join(lines)


def generate_pointwise_mul(n: int) -> str:
    """
    Generate an n-point pointwise complex multiply kernel.

    Uses NEON .2d vector operations (ld2/st2 with .2d IS valid) to process
    two complex numbers per iteration.

    Calling convention:
    - x0: pointer A (AoS complex doubles)
    - x1: pointer B (AoS complex doubles)
    - x2: pointer out (AoS complex doubles), or 0 for in-place on A
    - x0 return: checksum from out[0]
    """
    _validate_supported_size(n)
    vector_iters = n // 2
    out_ok = f"L_pwmul_{n}_out_ok"
    loop = f"L_pwmul_{n}_loop"

    lines: list[str] = [
        ".text",
        ".p2align 4",
        f".globl _pointwise_mul_{n}",
        f"_pointwise_mul_{n}:",
        "    mov x9, x0",
        "    mov x10, x1",
        f"    cbnz x2, {out_ok}",
        "    mov x2, x0",
        f"{out_ok}:",
        "    mov x11, x2",
    ]
    lines.extend(f"    {inst}" for inst in _mov_imm("x12", vector_iters))
    lines.append(f"{loop}:")
    lines.append("    prfm pldl1keep, [x9, #256]")
    lines.append("    prfm pldl1keep, [x10, #256]")
    lines.append("    ld2 {v0.2d, v1.2d}, [x9], #32")
    lines.append("    ld2 {v2.2d, v3.2d}, [x10], #32")
    lines.extend(f"    {inst}" for inst in _emit_cmul_v2("v0", "v1", "v2", "v3", "v4", "v5"))
    lines.append("    st2 {v4.2d, v5.2d}, [x11], #32")
    lines.append("    subs x12, x12, #1")
    lines.append(f"    b.ne {loop}")
    lines.extend(f"    {inst}" for inst in _emit_return_checksum("x2"))
    lines.append("    ret")
    return "\n".join(lines)


def generate_benchmark_harness(n: int, include_inverse: bool = True) -> str:
    """
    Generate benchmark harness for API execution.

    Exposes:
    - _bench_fft_{n}: initialize data, run forward FFT, return checksum
    - _bench_ifft_{n}: initialize data, run forward+inverse, return checksum (optional)
    - _user_entry: default entrypoint -> _bench_fft_{n}
    """
    _validate_supported_size(n)
    init_fn = f"_init_bench_data_{n}"
    init_loop = f"L_init_bench_data_{n}"

    lines: list[str] = [
        ".data",
        ".p2align 7",
        f"_bench_data_{n}:",
        f"    .zero {n * 16}",
        ".text",
        ".p2align 4",
        f"{init_fn}:",
        "    mov x10, x0",
    ]
    lines.extend(f"    {inst}" for inst in _mov_imm("x9", 0))
    lines.extend(f"    {inst}" for inst in _mov_imm("x11", n))
    lines.append(f"{init_loop}:")
    lines.append("    and x12, x9, #15")
    lines.append("    sub x12, x12, #8")
    lines.append("    scvtf d0, x12")
    lines.append("    and x13, x9, #31")
    lines.append("    sub x13, x13, #16")
    lines.append("    scvtf d1, x13")
    lines.append("    stp d0, d1, [x10], #16")
    lines.append("    add x9, x9, #1")
    lines.append("    subs x11, x11, #1")
    lines.append(f"    b.ne {init_loop}")
    lines.append("    ret")

    lines.extend(
        [
            ".p2align 4",
            f".globl _bench_fft_{n}",
            f"_bench_fft_{n}:",
            "    stp x29, x30, [sp, #-32]!",
            "    mov x29, sp",
            "    stp x19, x20, [sp, #16]",
        ]
    )
    lines.extend(f"    {inst}" for inst in _emit_label_address("x19", f"_bench_data_{n}"))
    lines.append("    mov x0, x19")
    lines.append(f"    bl {init_fn}")
    lines.append("    mov x0, x19")
    lines.append(f"    bl _fft_fwd_{n}")
    lines.append("    ldp x19, x20, [sp, #16]")
    lines.append("    ldp x29, x30, [sp], #32")
    lines.append("    ret")

    if include_inverse:
        lines.extend(
            [
                ".p2align 4",
                f".globl _bench_ifft_{n}",
                f"_bench_ifft_{n}:",
                "    stp x29, x30, [sp, #-32]!",
                "    mov x29, sp",
                "    stp x19, x20, [sp, #16]",
            ]
        )
        lines.extend(f"    {inst}" for inst in _emit_label_address("x19", f"_bench_data_{n}"))
        lines.append("    mov x0, x19")
        lines.append(f"    bl {init_fn}")
        lines.append("    mov x0, x19")
        lines.append(f"    bl _fft_fwd_{n}")
        lines.append("    mov x0, x19")
        lines.append(f"    bl _fft_inv_{n}")
        lines.append("    ldp x19, x20, [sp, #16]")
        lines.append("    ldp x29, x30, [sp], #32")
        lines.append("    ret")

    lines.extend(
        [
            ".p2align 4",
            ".globl _user_entry",
            "_user_entry:",
            f"    b _bench_fft_{n}",
        ]
    )
    return "\n".join(lines)


def generate_program(n: int, include_inverse: bool = True, include_pointwise: bool = True) -> str:
    """
    Generate full standalone assembly source for one FFT size.

    The source includes FFT kernels, pointwise multiply, benchmark entrypoints,
    and twiddle/constant data tables.
    """
    _validate_supported_size(n)

    parts = [
        f"// Generated ARM64 radix-4 FFT program for N={n}",
        generate_fft_forward(n),
    ]
    if include_inverse:
        parts.append(generate_fft_inverse(n))
    if include_pointwise:
        parts.append(generate_pointwise_mul(n))
    parts.append(generate_benchmark_harness(n, include_inverse=include_inverse))
    parts.append(generate_twiddles(n))

    source = "\n\n".join(parts) + "\n"
    if len(source) > MAX_SOURCE_CHARS:
        raise ValueError(
            f"Generated source for N={n} is {len(source)} chars, "
            f"exceeding the {MAX_SOURCE_CHARS} char limit."
        )
    return source


def generate_sources_for_sizes(sizes: Iterable[int] = SUPPORTED_SIZES) -> dict[int, str]:
    """Generate full assembly source for each requested FFT size."""
    return {n: generate_program(n) for n in sizes}


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for writing generated assembly to disk."""
    parser = argparse.ArgumentParser(description="Generate ARM64 radix-4 FFT assembly.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1024],
        choices=SUPPORTED_SIZES,
        help="FFT sizes to generate.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write .S files. If omitted, prints source to stdout.",
    )
    parser.add_argument(
        "--no-inverse",
        action="store_true",
        help="Skip inverse FFT generation.",
    )
    parser.add_argument(
        "--no-pointwise",
        action="store_true",
        help="Skip pointwise multiply kernel generation.",
    )
    args = parser.parse_args(argv)

    for n in args.sizes:
        source = generate_program(
            n=n,
            include_inverse=not args.no_inverse,
            include_pointwise=not args.no_pointwise,
        )
        if args.out_dir is None:
            print(source)
        else:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / f"fft_{n}.S"
            out_path.write_text(source, encoding="utf-8")
            print(f"Wrote {out_path} ({len(source)} chars)")


if __name__ == "__main__":
    main()