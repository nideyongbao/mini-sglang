"""H20 image compatibility: prefer installed FA3 over incompatible FA4/CUTLASS.

Run only in the disposable benchmark container. The original is retained next
to the patched file; no model equations or weights are changed.
"""

from pathlib import Path

p = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py")
s = p.read_text()
marker = "    try:\n        from flash_attn.cute.interface import flash_attn_varlen_func as cute_varlen_func"
if "MINISGL_BENCH_FA3" not in s:
    assert marker in s
    p.with_suffix(".py.original").write_text(s)
    s = s.replace(
        marker,
        "    # MINISGL_BENCH_FA3: existing Hopper-compatible kernel.\n    from sgl_kernel.flash_attn import flash_attn_varlen_func\n    return flash_attn_varlen_func\n\n"
        + marker,
    )
    p.write_text(s)
print("Reference QSA packed attention uses sgl_kernel FA3")
