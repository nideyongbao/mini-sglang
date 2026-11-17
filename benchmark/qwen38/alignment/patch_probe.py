from pathlib import Path

p = Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py")
s = p.read_text()
if "# QWEN38_ALIGNMENT_PROBE" not in s:
    p.write_text(
        s
        + '\n# QWEN38_ALIGNMENT_PROBE: only active in the disposable diagnostic container.\nimport os as _qwen38_probe_os\nif _qwen38_probe_os.environ.get("QWEN38_TRACE_DIR"):\n    from probe import install_sglang as _qwen38_install_probe\n    _qwen38_install_probe()\n'
    )
