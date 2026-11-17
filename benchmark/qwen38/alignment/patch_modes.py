"""Explicit reference-only deterministic QSA policy; independent of tracing."""

from pathlib import Path

p = Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py")
s = p.read_text()
if "# QWEN38_REFERENCE_MODE" not in s:
    s += """
# QWEN38_REFERENCE_MODE: enable only for the controlled reference comparison.
import os as _q38_mode_os
if _q38_mode_os.environ.get("MINISGL_QWEN38_REFERENCE_MODE") == "strict":
    from minisgl.models.qwen3_8_reference import install_reference_policy
    install_reference_policy()
"""
    p.write_text(s)
s = p.read_text()
if "# QWEN38_MODE_LOGITS" not in s:
    p.write_text(s + """
# QWEN38_MODE_LOGITS: opt-in post-graph precision probe, never used for timing.
if _q38_mode_os.environ.get("QWEN38_MODE_PROBE_DIR"):
    from mode_probe import install_reference
    install_reference()
""")
s = p.read_text()
if "# QWEN38_FIXED_CONTINUATION" not in s:
    p.write_text(s + """
# QWEN38_FIXED_CONTINUATION: same generated input history during controlled timing.
if _q38_mode_os.environ.get("QWEN38_FIXED_CONTINUATION") == "1":
    from mode_probe import install_fixed_continuation_reference
    install_fixed_continuation_reference()
""")
