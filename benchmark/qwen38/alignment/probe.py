"""Opt-in eager tensor capture. Never enable during performance measurements."""

import json
import os
from pathlib import Path

import torch

CURRENT = None


def first(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        return first(x[0])
    return getattr(x, "hidden_states", None)


def save(name, x):
    if CURRENT is not None and isinstance(x, torch.Tensor):
        CURRENT["tensors"][name] = x.detach().clone().cpu()


def wrap(obj, method, name, normalize=False, input_name=None):
    original = getattr(obj, method)

    def call(*args, **kwargs):
        if input_name and args:
            save(input_name, first(args[0]))
        if name.endswith(".core"):
            for k in ("mixed_qkv", "a", "b"):
                save(name + "." + k, kwargs.get(k))
        if name.endswith(".qsa.core"):
            for k, v in zip(("q", "k", "v"), args[:3]):
                save(name + "." + k, v)
        out = original(*args, **kwargs)
        if normalize:
            save(name + ".input", args[0])
            save(name + ".norm", out[1][1])
        value = first(out)
        if name.endswith(".core") and value is not None:
            value = value.reshape(-1, value.shape[-2] * value.shape[-1])
        save(name, value)
        if CURRENT is not None and name.endswith(".qsa.qkv"):
            CURRENT["_qsa_stem"] = name.rsplit(".", 1)[0]
        return out

    setattr(obj, method, call)


def install_deterministic_topk():
    if os.environ.get("QWEN38_STABLE_TOPK") != "1":
        return
    import sglang.kernels.ops.elementwise.fast_topk as module

    def stable(logits, lengths, topk, row_starts=None):
        starts = torch.zeros_like(lengths) if row_starts is None else row_starts
        columns = torch.arange(logits.shape[1], device=logits.device)[None]
        valid = (columns >= starts[:, None]) & (columns < (starts + lengths)[:, None])
        scores = logits.masked_fill(~valid, -float("inf"))
        order = scores.argsort(dim=-1, descending=True, stable=True)[:, :topk]
        ids = order - starts[:, None]
        ranks = torch.arange(order.shape[1], device=logits.device)[None]
        ids = torch.where(ranks < lengths[:, None], ids, -1).int()
        if ids.shape[1] < topk:
            ids = torch.nn.functional.pad(ids, (0, topk - ids.shape[1]), value=-1)
        ranks = torch.arange(topk, device=logits.device)[None].expand_as(ids)
        all_ids = torch.where(ranks < lengths[:, None], ranks, -1).int()
        ids = torch.where((lengths <= topk)[:, None], all_ids, ids)
        return ids.contiguous()

    module.fast_topk = stable


def attach(model, backend):
    install_scores(backend)
    install_deterministic_topk()
    trunk = model.model
    wrap(trunk.embed_tokens, "forward", "embedding")
    layers = trunk.layers.op_list if backend == "mini" else trunk.layers
    for i, layer in enumerate(layers):
        stem = f"layer.{i}"
        for attr, label in [
            ("attn_hyper_connection", "attn_hc"),
            ("mlp_hyper_connection", "mlp_hc"),
        ]:
            hc = getattr(layer, attr)
            wrap(hc, "mix", stem + "." + label + ".mix", normalize=True)
            wrap(hc, "combine", stem + "." + label + ".combine")
        if hasattr(layer, "linear_attn"):
            gdn = layer.linear_attn
            wrap(gdn.attn, "forward", stem + ".gdn.core")
            wrap(gdn.out_proj, "forward", stem + ".gdn.out", input_name=stem + ".gdn.gated")
            wrap(gdn, "forward", stem + ".attention")
        else:
            attn = layer.self_attn if backend == "mini" else layer
            wrap(attn.qkv_proj, "forward", stem + ".qsa.qkv")
            wrap(attn.o_proj, "forward", stem + ".attention", input_name=stem + ".qsa.gated")
            wrap(attn.indexer.index_qk_proj, "forward", stem + ".qsa.index_qk")
            if backend == "sglang":
                wrap(attn.attn, "forward", stem + ".qsa.core")
            original_index = attn.indexer.forward

            def index(*args, _original=original_index, _stem=stem, **kwargs):
                if CURRENT is not None:
                    CURRENT["_qsa_stem"] = _stem + ".qsa"
                out = _original(*args, **kwargs)
                value = out
                if backend == "mini" and CURRENT is not None:
                    from minisgl.core import get_global_ctx

                    ctx = get_global_ctx()
                    parts, offset = [], 0
                    for req in ctx.batch.reqs:
                        n = 1 if ctx.batch.is_decode else req.extend_len
                        end = req.cached_len + n
                        slots = ctx.page_table[req.table_idx, :end].long()
                        inverse = torch.empty(
                            int(slots.max()) + 1, device=out.device, dtype=torch.int32
                        )
                        inverse[slots] = torch.arange(end, device=out.device, dtype=torch.int32)
                        part = out[offset : offset + n]
                        parts.append(torch.where(part >= 0, inverse[part.clamp_min(0).long()], -1))
                        offset += n
                    value = torch.cat(parts)
                save(_stem + ".qsa.indices", value)
                return out

            attn.indexer.forward = index
            if backend == "sglang":
                original_select = attn.indexer.select_decode_tokens

                def select(*args, _original=original_select, _stem=stem, **kwargs):
                    if CURRENT is not None:
                        q, cache, table, lengths = args[:4]
                        save(_stem + ".qsa.index.q", q[:, :4])
                        n = int(lengths.max())
                        pages = cache[table.long().clamp_min(0)].flatten(1, 2)
                        save(_stem + ".qsa.index.k", pages[:, :n].squeeze(-2))
                    return _original(*args, **kwargs)

                attn.indexer.select_decode_tokens = select
            else:
                original_score = attn.indexer.decode_scores

                def scores(q, keys, positions, _original=original_score, _stem=stem):
                    if CURRENT is not None:
                        n = int(((positions + 1) // 4).max())
                        save(_stem + ".qsa.index.q", q)
                        save(_stem + ".qsa.index.k", keys[:, :n])
                    return _original(q, keys, positions)

                attn.indexer.decode_scores = scores
        wrap(layer.mlp.gate, "forward", stem + ".router")
        wrap(layer.mlp.experts, "forward", stem + ".experts")
        wrap(layer.mlp.shared_expert, "forward", stem + ".shared")
        wrap(layer.mlp, "forward", stem + ".mlp")
        if getattr(layer, "ple", None) is not None:
            ple = layer.ple
            wrap(ple.ple_embedding, "forward", stem + ".ple.embedding")
            for name in ["key_proj", "value_proj", "norm_key", "norm_query", "norm_conv"]:
                wrap(getattr(ple, name), "forward", stem + ".ple." + name)
            wrap(ple, "forward", stem + ".ple")
        wrap(layer, "forward", stem + ".output")
    wrap(trunk.hyper_connection_mixer, "mix", "final_hc.mix", normalize=True)


def install_scores(backend):
    if backend == "sglang":
        import sglang.srt.layers.attention.qsa.qsa_indexer as score_module
    else:
        import sglang.srt.layers.attention.qsa.mqa as score_module
    original_score_fn = score_module.qsa_mqa_decode

    def score_fn(q, cache, table, lengths, maxlen, *args, **kwargs):
        out = original_score_fn(q, cache, table, lengths, maxlen, *args, **kwargs)
        if CURRENT is not None:
            save(CURRENT["_qsa_stem"] + ".index.scores", out[:, : int(lengths.max())])
            CURRENT["index_score_width"] = out.shape[-1]
        return out

    score_module.qsa_mqa_decode = score_fn


def install_root(cls, backend):
    original = cls.forward

    def call(self, *args, **kwargs):
        global CURRENT
        directory = os.environ.get("QWEN38_TRACE_DIR")
        arm = Path(directory) / "armed.json" if directory else None
        if not arm or not arm.exists():
            return original(self, *args, **kwargs)
        spec = json.loads(arm.read_text())
        if not getattr(self, "_qwen38_probed", False):
            attach(self, backend)
            self._qwen38_probed = True
        if backend == "mini":
            from minisgl.core import get_global_ctx

            batch = get_global_ctx().batch
            ids, positions = batch.input_ids, batch.positions
            mode = "decode" if batch.is_decode else "prefill"
        else:
            ids = kwargs.get("input_ids", args[0] if args else None)
            positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
            fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
            mode = "decode" if fb.forward_mode.is_decode() else "prefill"
        positions_cpu = positions.detach().cpu()
        start, end = int(positions_cpu.min()), int(positions_cpu.max())
        CURRENT = dict(
            case=spec["case"],
            backend=backend,
            mode=mode,
            input_ids=ids.detach().cpu(),
            positions=positions_cpu,
            tensors={},
        )
        output = original(self, *args, **kwargs)
        logits = output if backend == "mini" else output.next_token_logits
        save("logits", logits)
        label = os.environ.get("QWEN38_TRACE_LABEL", "mini") if backend == "mini" else backend
        path = Path(directory) / spec["case"] / label
        path.mkdir(parents=True, exist_ok=True)
        torch.save(CURRENT, path / f"{mode}-{start:04d}-{end:04d}.pt")
        CURRENT = None
        # Save true logits first; only then force a common continuation.
        token = 1000 + max(0, end - (spec["prompt_length"] - 1))
        logits.fill_(-10000)
        logits[:, token] = 10000
        return output

    cls.forward = call


def install_sglang():
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

    install_root(Qwen4ExpForConditionalGeneration, "sglang")


def install_mini():
    from minisgl.models.qwen3_8 import Qwen38ForCausalLM

    install_root(Qwen38ForCausalLM, "mini")
    import minisgl.models.qwen3_8_qsa as module

    original_prepare = module.prepare_qk_gate

    def prepare(*args, **kwargs):
        out = original_prepare(*args, **kwargs)
        if CURRENT is not None:
            stem = CURRENT["_qsa_stem"] + ".core"
            save(stem + ".q", out[0])
            save(stem + ".k", out[1])
        return out

    module.prepare_qk_gate = prepare
    for name in ("sparse_attention", "sparse_decode"):
        original_fn = getattr(module, name)

        def sparse(*args, _original=original_fn, **kwargs):
            out = _original(*args, **kwargs)
            if CURRENT is not None:
                save(CURRENT["_qsa_stem"] + ".core", out)
            return out

        setattr(module, name, sparse)
