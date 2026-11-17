# Qwen3.8 Flash Next：适配、精度与性能报告

测试日期：2026-11-17；分支：`feat/qwen38-next-local`；基线提交：`b32c62a62b895d38785d7ac23089cc413e3684de`。

## 结论

已实现目标 tiny FP8 模型的 Mini-SGLang 原生纯文本部署，覆盖 GDN、QSA、HyperConnection、PLE 和 FP8 MoE。不是调用 SGLang Engine 的服务包装。部署步骤见 [使用说明](qwen38-next-local.md)。

独占 H20 实测，输出 128 tokens 的离线 12 组中，Mini-SGLang 吞吐为参考 SGLang 的 **1.17–2.05 倍**，几何平均 **1.51 倍**。原生 HTTP 六组吞吐比为 **1.33–2.31 倍**。这些结论仅适用于本次随机 tiny checkpoint、当前镜像和指定配置。

数值误差、严格 token 一致率和并列情况分别列于下方；不将近似数值一致表述为逐 token 完全相同。

## 环境与可复现性

| 项目 | 值 |
|---|---|
| GPU | 1 × NVIDIA H20，97871 MiB，驱动 570.195.03 |
| CPU | Intel Xeon Platinum 8469C，24 logical CPUs |
| 镜像 | `lmsysorg/sglang:dev-cu12-qwen38-next-local` |
| 镜像 ID | `sha256:51eb4a3747ac86e2c46ad925302fc38aa3fe0095728e9dd90654f0d266205145` |
| SGLang 源码 | `4ccff141dbe992794f9da6c3aa23535b4f72000d` |
| 软件 | PyTorch 2.13.0+cu129 / Transformers 5.12.1 / Triton 3.7.1 |
| 权重路径 | `/shard_data/brooksli/models/tiny-random/qwen3.8-flash-next-fp8` |
| 模型配置 | 8 层，hidden=32，Q heads=12、KV heads=1、head_dim=32，vocab=248320 |
| 混合结构 | 6 GDN + 2 QSA；HC=4/rank=4；第 2 层 PLE，ngram=3 |
| MoE / QSA | 32 experts/top-10，FP8 block=32×32；QSA compress=4、budget=2048 |

[模型文件 SHA256 清单](../benchmark/qwen38/results/2026-11-17/model-manifest.json)记录配置、tokenizer 和权重。模型文件、宿主镜像未被修改。

参考端须执行 `benchmark/qwen38/patch_reference.py`：当前镜像的 QSA decode 会进入不兼容的 FA4/CUTLASS 路径，临时容器内改用已安装的 FA3。参考端同时使用 `--language-only`、精确 graph batch 1–8、关闭 graph padding，避开本机观察到的部分 padding 批次非有限 logprob。所有基准和精度数据均以这份经过兼容修正的参考实现为准。

## 适配实现与回归

- 注册 `Qwen4ExpForConditionalGeneration`，保留其完整 text/quantization config；特殊加载器保留 FP8 权重、FP32 scale 和 int64 PLE 哈希，严格校验加载形状。
- GDN 使用 sigmoid 输出门；扩展 head_dim=32 的 RoPE 和 64-byte embedding/KV 拷贝。
- 实现 HC 输入混合/残差注入与 tiny 形状的融合 Triton kernel。
- 实现 PLE 的 EOS-aware n-gram 哈希、数字顺序 embedding 分片拼接、跨 chunk 卷积状态；槽位复用时重置历史。
- 实现 QSA 压缩 key、因果 top-k 块和当前块尾部，稀疏预填充与 FA3 解码；上下文不超过 budget 时采用等价 dense 路径。
- FP8 专家调用底层 Triton GEMM；GDN graph replay 的状态索引改为一次 pinned 异步拷贝，减少逐槽位同步。

最终 GPU 回归 **16 passed**，覆盖新增 attention/HC/norm/gate/窄宽度拷贝和已有 Qwen3.5 fused/attention-backend 用例。原生 OpenAI chat 接口也完成并发请求和整套 HTTP 性能测试。

## 精度

| 场景 | 自由生成 token 一致 | 整序列一致 | Teacher top-1 | 命中参考最高概率并列集合 | top-20 最大 / 平均绝对误差 |
|---|---:|---:|---:|---:|---:|
| 主：短上下文，chunk=512，graph 1–8 | 188/224 | 10/14 | 70/70 | 70/70 | 0.00195503 / 0.00025277 |
| 主：长上下文，chunk=512，graph 1–8 | 79/96 | 3/6 | 28/30 | 30/30 | 0.00195503 / 0.00027264 |
| 补充：短上下文，chunk=127，graph 1/2/4/8 | 188/224 | 10/14 | 70/70 | 70/70 | 0.00195599 / 0.00025480 |
| 补充：长上下文，chunk=512，graph 1/2/4/8 | 79/96 | 2/6 | 28/30 | 29/30 | 0.00098133 / 0.00025646 |

方法：固定输入 token、temperature=0、ignore_eos=True，每个 prompt 生成 16 tokens。Teacher forcing 沿参考输出历史，在 step 0/1/3/7/15 比较下一 token；logprob 误差仅统计参考 top-20 集合，**不是全词表 logits 的误差界**。默认绝对误差容限 0.005 为本实验回归阈值；四组均通过，最大误差 0.00195599。主测试 teacher-forced 严格 top-1 为 98/100，参考并列集合命中为 100/100。

主测试覆盖输入长度 1–4000、512/2048 边界、EOS、重复 token 和重复请求槽位使用；补充 chunk=127（非 QSA 压缩比整数倍）及 CLI graph 档位 1/2/4/8 的 padding 路径。

在长上下文主测试的两个 teacher-forced top-1 差异中，双方 token 在参考端的 logprob 完全并列。自由生成的长上下文首次分歧包含精确并列，也包含参考概率间隔 0.0009765625 或 0.001953125 的近似并列；后续历史不同会放大 token 序列差异。因此只能判断本次数值误差较小，不能宣称所有生成序列严格一致。随机权重不具备真实模型语义质量评估意义。

## 性能方法

取得用户授权后停止已有 `sglang-qwen38-dflash2`，测试开始时 GPU 显存占用 **0 MiB**、利用率 **0%**。SGLang 与 Mini-SGLang 串行运行，原有监控容器保留。共享 GPU 的调试数据不进入下列表格。没有锁定 GPU 时钟，也没有进行长期负载或统计显著性检验。

两边统一 TP=1、BF16 激活/attention KV、FP8 专家、上下文 4096、KV token 容量 32768、最多 8 个请求、chunk=512、禁用前缀复用、greedy、固定输出长度。底层分页实现保留各框架配置：参考 QSA 自动使用 page_size=64，Mini-SGLang 为 1；这是完整框架比较，不是单 kernel 测试。

离线调用分别为 `sglang.Engine.generate` 和 `minisgl.llm.LLM.generate`，固定相同随机 token 输入；CUDA 同步后计时。每组预热 2 次、记录 5 次，使用中位数，排除加载、JIT 和 graph capture。两边离线 graph 都捕获 batch 1–8。输出 1 token 的时间含预填充和框架开销，属于离线 time-to-one-token 近似值，不是在线流式 TTFT。

### 离线：输出 128 tokens

吞吐为整批输出 tokens / 整批耗时；倍率 = Mini 吞吐 / SGLang 吞吐。

| 输入 tokens | 并发 | SGLang ms | Mini ms | SGLang tok/s | Mini tok/s | 倍率 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 174.60 | 128.27 | 733.1 | 997.9 | 1.36× |
| 128 | 4 | 176.01 | 139.39 | 2908.9 | 3673.1 | 1.26× |
| 128 | 8 | 203.13 | 173.00 | 5041.0 | 5918.9 | 1.17× |
| 512 | 1 | 172.37 | 132.45 | 742.6 | 966.4 | 1.30× |
| 512 | 4 | 258.79 | 179.83 | 1978.4 | 2847.2 | 1.44× |
| 512 | 8 | 371.95 | 249.57 | 2753.0 | 4103.0 | 1.49× |
| 1536 | 1 | 229.65 | 167.33 | 557.4 | 765.0 | 1.37× |
| 1536 | 4 | 483.93 | 296.82 | 1058.0 | 1724.9 | 1.63× |
| 1536 | 8 | 966.10 | 471.83 | 1059.9 | 2170.3 | 2.05× |
| 3072 | 1 | 314.00 | 210.85 | 407.6 | 607.1 | 1.49× |
| 3072 | 4 | 822.62 | 458.20 | 622.4 | 1117.4 | 1.80× |
| 3072 | 8 | 1645.36 | 805.98 | 622.4 | 1270.5 | 2.04× |

### 离线：输出 1 token

| 输入 tokens | 并发 | SGLang ms | Mini ms | 延迟加速比 |
|---:|---:|---:|---:|---:|
| 128 | 1 | 30.10 | 13.48 | 2.23× |
| 128 | 4 | 31.26 | 17.99 | 1.74× |
| 128 | 8 | 60.39 | 35.96 | 1.68× |
| 512 | 1 | 29.64 | 13.79 | 2.15× |
| 512 | 4 | 115.07 | 54.38 | 2.12× |
| 512 | 8 | 227.96 | 109.60 | 2.08× |
| 1536 | 1 | 86.06 | 40.67 | 2.12× |
| 1536 | 4 | 345.27 | 162.07 | 2.13× |
| 1536 | 8 | 671.85 | 325.54 | 2.06× |
| 3072 | 1 | 170.88 | 82.42 | 2.07× |
| 3072 | 4 | 676.11 | 323.56 | 2.09× |
| 3072 | 8 | 1337.65 | 648.95 | 2.06× |

### HTTP：输出 128 tokens

两个框架分别启动自己的原生服务；loopback HTTP 客户端同时发送 1/4/8 个 `/v1/chat/completions` 请求。输入为相同重复文本，关闭 thinking；两边实际模板后 token 数都为 140/2060。每组预热 2 次、测量 5 次。表中为整批完成的 E2E 耗时，包含排队、tokenization、网络与 detokenization；不是流式 TTFT/ITL。客户端和服务端位于同一台机器。

Mini CLI 使用原生 graph 档位 1/2/4/8，参考使用精确 batch 1–8（为避开其 padding 问题）；原始 JSON 还保留逐请求耗时和 usage。这里只比较指定并发的 burst，不代表饱和流量容量。

| 实际输入 tokens | 并发 | SGLang ms | Mini ms | SGLang tok/s | Mini tok/s | 倍率 |
|---:|---:|---:|---:|---:|---:|---:|
| 140 | 1 | 174.51 | 131.32 | 733.5 | 974.7 | 1.33× |
| 140 | 4 | 240.20 | 158.78 | 2131.6 | 3224.7 | 1.51× |
| 140 | 8 | 442.39 | 191.46 | 2314.7 | 5348.5 | 2.31× |
| 2060 | 1 | 299.57 | 199.98 | 427.3 | 640.1 | 1.50× |
| 2060 | 4 | 640.96 | 380.84 | 798.8 | 1344.4 | 1.68× |
| 2060 | 8 | 1276.37 | 612.91 | 802.3 | 1670.7 | 2.08× |

## 已部署服务对测量的影响与恢复

- 同卡服务有请求：会争用 SM、显存带宽、GPU 调度和 CPU，吞吐与尾延迟可能变化，不适合直接归因于框架差异。
- 同卡服务无请求：权重/KV/graph 显存通常仍驻留，影响可用容量；还需检查后台 GPU 工作、上下文切换、功耗和时钟。不能仅凭“没有业务请求”认定是独占环境。
- 本次正式性能表均来自停服后的独占测量；数值回归允许共享 GPU，但其耗时未用于性能结论。

原服务使用 `--rm`，停止后容器会删除，因此恢复通过原脚本 `/shard_data/brooksli/workspace/0826/deploy.sh` 重建同名 `sglang-qwen38-dflash2`。已比较停服前与恢复后的完整 `ServerArgs`，差异为空；最终 `/health`、`/v1/models`、`/metrics` 均返回 200，模型别名为 `qwen`，见 [恢复记录](../benchmark/qwen38/results/2026-11-17/service-restoration.json)。保留原 DFlash2、端口 8000、模型别名 qwen、metrics 配置和已有监控。首次尝试遇到离线参数名及自动删除容器问题，修正后重新完成全部正式测量；首次尝试数据未进入结果表。

## 补充：两条模块/算子路径为什么不同

本节为 2026-11-17 对已测代码、镜像内源码和原始结果的复核。**属于源码分派审计及已有测量的算术分析，不是新增的逐算子 GPU profile 或消融实验。** 本次未启动测试 GPU 服务、未停止现有业务服务。[分析数据及参考源码哈希](../benchmark/qwen38/results/2026-11-17/path-analysis.json)可用于核对。

### 模型结构相同，执行实现不同

两边都是 embedding → 8 层（6 GDN / 2 QSA，各带 HC 和 MoE，第 2 层带 PLE）→ 最终 HC mixer → LM head。均保留 FP8 专家、QSA 稀疏选择和 PLE；性能测试使用 context=4096，**没有命中 mini 的 ≤2048 dense 等价捷径**。没有减少层数、专家 top-k 或输出长度，也没有启用推测解码。

| 模块 | 当前 SGLang 路径 | 当前 Mini-SGLang 路径 | 关键差异与边界 |
|---|---|---|---|
| HC mix，逻辑上 17 次/forward | `GroupedGemmaRMSNorm` → `_mix_compute`（torch.compile 的 Linear/SiLU/Linear/sigmoid/mul/mean） | `tiny_hc_mix`：group norm、两次低秩投影、gate、mean 融为一个 Triton kernel | hidden=32、HC×hidden=128 不满足参考融合核的形状条件；mini 特化本次 4×32/rank=4 |
| HC combine，16 次/forward | `_combine_compute`：Linear + gate + residual，局部 torch.compile | `tiny_hc_combine` 单个 Triton kernel | 参考 JIT combine 要求总宽度可被 2048 整除，本例不满足；数值归约与中间舍入可能不同 |
| PLE 分组归一化，3 次/forward | group_size=32 不满足参考 JIT 的 512 整除条件，走 PyTorch FP32 mean/rsqrt/mul/cast 链 | 自定义 `grouped_norm` Triton kernel | 不改变 RMSNorm 公式；减少中间张量和 kernel 分拆 |
| GDN 输入投影 | BF16 且无 LoRA 时装载后合并 QKVZ/BA；M≤1024 可走一次 `bf16_gemm_dispatch` | `in_proj_qkvz` 和 `in_proj_ba` 两次 Linear，再做 fused layout preparation | **这一项参考端反而融合得更多**，不能概括为 mini 的所有层都更快 |
| GDN prefill，6 层 | `causal_conv1d_fn` → Q/K/V split → `fused_gdn_gating` → `chunk_gated_delta_rule` | 复用同一 SGLang causal-conv、gating 和 chunk GDN 核心 | 核心算法/内核家族相同，metadata、布局准备、外围 norm/gate 不完全相同；两边 SSM state 都是 FP32 |
| GDN decode | `causal_conv1d_update` → `fused_recurrent_gated_delta_rule_packed_decode` → gated RMSNorm(sigmoid) | 本地 causal-conv update → `gdn_decode.packed_decode` → `gated_rmsnorm_(sigmoid)` | 同样的 recurrent 公式，但本地独立 Triton 实现，不能声称 binary/kernel 完全相同 |
| QSA QKV/Norm/RoPE | QKV projection → `fused_qk_gemma_rmsnorm_rope_gate` → output fused sigmoid-mul | QKV projection → `prepare_qk_gate` → `apply_attention_gate_` | 双方都有融合准备和输出门，不是“参考全部 eager、mini 全部 fused” |
| QSA indexer | fused prep/compress 条件分派；CUDA+TileLang 可用时使用 TileLang MQA score；fast top-k/块展开 | PyTorch FP32 `einsum` + ReLU/head-sum；复用参考 `qsa_fast_topk` / `expand_qsa_block_indices` | score、norm/rope、压缩和归约实现不同；不能保证选择块索引逐项相同。这里只审计源码分派，未新增捕获运行时全部 indexer 中间值 |
| QSA prefill，2 层 | 首 chunk：`sparse_gqa_fwd_interface_triton`；后续 chunk：先收集全上下文 KV，再 `sparse_gqa_fwd_interface_triton_ck` | 页表映射物理位置后，`qsa._sparse` 直接读取分页 KV | 后续 chunk 的 gather/cat 和 host readback 是明确差异；attention online-softmax 思路相同，tile/循环/掩码细节不同 |
| QSA decode | valid-count/cu-seqlens + compact KV gather → patched FA3 varlen | length/cumsum + `_gather` → `flash_attn_varlen_func(ver=3)` | 最终 attention 同为 FA3，准备和 gather 路径不同；不是拿 mini FA3 对参考 FA4 做性能对比 |
| Routed MoE，8 层 | TopK/FusedMoE runner → align → 2 次 `invoke_fused_moe_kernel` + activation + reduce | mini top-k/align → 同一 `invoke_fused_moe_kernel` 两次 + activation + reduce | 同为 dynamic block-FP8 W8A8；当前默认 tile 配置相同，不能把优势归因于换了精度更低的 GEMM |
| Shared expert | BF16 gated MLP + sigmoid shared gate | BF16 gated MLP + sigmoid shared gate | 模型功能相同，包装/elementwise fusion 不同 |
| PLE embedding/conv | FP8 table gather 输出 BF16，再乘 scale；状态管理支持更广泛 batch/缓存场景；部分 decode 有融合快路 | 加载时缓存 FP8→BF16 table，lookup 后乘原 scale；请求槽位历史 + PyTorch depthwise conv | FP8→BF16 本身可精确表示；此处存储差异不是额外 FP8 量化损失。参考端也有 mini 未实现的 PLE 优化 |
| LM head / greedy | 选最后 hidden → BF16 matmul → logits 转 FP32 → `torch.argmax` | 选最后 hidden → BF16 `F.linear` → `torch.argmax` | BF16→FP32 转换本身不改变已有数值次序。两边 greedy 规则一致，差异不能简单归因于随机采样/不同 tie-break |
| 离线调用链 | `Engine.generate` → TokenizerManager → scheduler 子进程 → 返回结果 | `LLM.generate` 直接驱动本进程 Scheduler | 测的是 API E2E，框架/IPC 开销在内；不是纯 CUDA model.forward 对比 |

上表次数是模型每次 forward 的逻辑调用次数，不是 GPU trace 中精确的总 kernel launch 数；torch.compile 的分拆取决于形状和编译结果。

### 为什么 tiny 模型容易出现较大的比例差距

1. **HC/Norm 的参考快路没有覆盖本次小形状。** HC group=32、总宽=128；参考 grouped norm 要求 group_size 为 512 的倍数，HC fused mix/combine 要求总宽为 2048 的倍数；部分 CuTe mix 还要求 SM100。mini 针对 tiny 尺寸将 17 次 mix（含 norm）和 16 次 combine 各做成单核。这是源码可确认的差异；精确节省了多少毫秒尚无同条件消融数据。
2. **QSA chunk-prefill 存在额外数据搬运和同步。** 参考 `forward_extend` 对已有 prefix 的请求执行 `req_pool_indices.tolist()`、完整上下文 K/V 的 `index_select`/`cat`；随后 chunk kernel wrapper 用 `int((cu_q[1:] - cu_q[:-1]).max().item())` 读取 GPU scalar 来决定 launch grid。mini 直接按物理页表读取 KV、用已知 token 数决定 grid。对 hidden=32 的小模型，这类固定开销可能占较大比例；并非 attention 数学工作量减半。
3. **参考端并不是所有算子都慢。** GDN 投影合并、QSA indexer 专用 CUDA/TileLang 路径、PLE decode 融合等都可能有利于参考端。MoE 日志提示 tiny 形状缺少调优配置，但代码核对显示它回退的 block-FP8 配置正好与 mini 一致：M=64/N=32/K=32、GROUP_SIZE_M=32、warps=4、stages=3。所以“缺少 MoE 调优文件”不能独立解释本次加速。
4. **HTTP 比值还混入了调度和请求到达方式。** HTTP 请求经过独立线程并发提交、动态组 batch、队列和 detokenizer；两边 graph 档位也不同。离线/HTTP 输入数据不同（随机 token vs 重复文本），token 数也不同，不能把 HTTP 比值减去离线比值当成纯网络层优化。

已有时间数据可做以下辅助分解：

| 输入/并发 | 1-token：SGLang / mini ms | 128-token 减 1-token：SGLang / mini ms |
|---|---:|---:|
| 128 / 1 | 30.10 / 13.48 | 144.50 / 114.79 |
| 128 / 8 | 60.39 / 35.96 | 142.75 / 137.05 |
| 3072 / 8 | 1337.65 / 648.95 | 307.71 / 157.03 |

例如 3072/8 的总耗时差约 839.38 ms，其中 1-token 测量差约 688.70 ms，算术占比约 82%。这表明巨大差距已出现在预填充及框架开销所在阶段。但两次独立测量的中位数相减不等于实测纯 decode 时间，输出长度改变还会影响调度；这个 82% **不是**逐核 profiler 给出的归因比例。

### 为什么精度不是 100%

模型公式相同不意味着逐位结果相同。本分支重新实现/融合了 HC、归一化、GDN decode、QSA prefill 和 indexer；FP32 归约树、BF16 中间落盘/舍入位置、FMA/指数实现、矩阵形状和 batch 组织都可能改变最终 logits。MoE router top-k 与 QSA top-k 也是离散选择点，小数值变化可能改变路由；**当前没有逐层捕获证据证明它们实际在哪个样本发生变化**。

确定的证据来自相同历史的输出比较。主测试长上下文 prompt=2、step=0：

| 候选 token | SGLang logprob | mini logprob |
|---:|---:|---:|
| 14948（SGLang 选择） | -12.20106125 | -12.20203590 |
| 88089（mini 选择） | -12.20106125 | -12.20105934 |

参考端两者并列，mini 中 88089 高出 0.0009765625。两边都是 argmax；输入分布变了，结果自然可能不同。不能把“命中参考最高概率并列集合”误读为“logits 完全相同”或“逐 token 已经 100% 对齐”。

主测试自由生成共有 7 条序列出现首次分歧：其中 5 条首次分歧的两个候选在参考端 logprob 并列，另 2 条参考间隔分别为 0.0009765625 和 0.001953125。分歧发生后，两边把不同 token 送入下一步，PLE n-gram、GDN 状态、QSA/KV 历史随之不同；后续 token 不再是在相同输入上比较。因此自由生成 267/320（83.4%）不能解释为“16.6% 的算子计算错误”，也不能因此忽略真实数值差异。

Teacher forcing 将历史对齐后，主测试严格 top-1=98/100，参考并列集合命中=100/100；但补充 padded-graph 长上下文是 28/30 和 29/30，说明**并非所有测试差异都能用参考端精确并列解释**。小模型随机权重、248320 词表以及 BF16 量化后的候选近似并列共同使 argmax 较敏感。这不是正式权重任务精度认证。

当前短上下文精度测试的 mini 上限为 2048，使用 QSA→dense 等价路径，而参考始终走 QSA；长上下文测试双方走 QSA。正式性能测试两边均为 4096，故上述短上下文捷径没有用于获得性能表中的优势。

### 已知与尚未证明的界限

- 已确认：两边具体模块分派、参考 tiny 形状融合限制、QSA 后续 chunk 的 gather/host readback、MoE 相同核心与默认 tile、两边 greedy 均为 argmax、首次输出分歧处的概率间隔。
- 尚未证明：第一个发生误差的中间张量属于哪层/哪一核、HC/QSA/调度分别贡献多少毫秒、MoE/QSA top-k 是否在每个分歧样本保持一致。因此不能给出“某一个算子造成全部精度差异”或“某项优化贡献了全部 2 倍加速”的结论。
- 若继续要求严格数值对齐，需要相同固定输入、相同 prefill/decode 和 batch 形状，逐层抓取 HC mix、GDN/QSA、router/top-k、PLE、最终 hidden 和完整 logits，再逐项换回参考核做消融。只强行改 argmax tie-break 不能修复上游数值差异。

参考代码位置（固定在报告所列镜像与 commit，以下路径相对于 SGLang `python/sglang/`）：`srt/layers/hyperconnection.py`、`srt/layers/hc_mix_triton.py`、`srt/layers/attention/qwen_sparse_attn_backend.py`、`srt/layers/attention/qsa/{sparse_attn,qsa_indexer,mqa}.py`、`srt/layers/attention/linear/kernels/gdn_triton.py`、`srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py`、`srt/layers/{sampler,logits_processor}.py`、`srt/entrypoints/engine.py`。mini 对应源码为 `models/qwen3_8*.py`、`kernel/triton/qwen38.py`、`kernel/triton/qsa.py`、`attention/gdn.py`、`kernel/triton/gdn_decode.py`、`engine/sample.py`。

## 限制与结果文件

支持范围为目标本地 tiny checkpoint 的 TP=1、纯文本、BF16 激活、block-FP8 路由专家。未验证 4096 以上上下文、其他 GPU、真实大模型 checkpoint、MTP/推测解码、视觉和多 GPU；不支持前缀 PLE/QSA 状态复用，强制 naive cache。PLE embedding 查表保存 BF16 副本并在 lookup 时应用原 scale，专家 GEMM 保留 FP8。

性能优势不能外推到 27B/更大模型或不同后端；这里只有 5 次重复，没有在线流式 TTFT/ITL/p99 或真实请求到达率曲线。全词表数值指标和真实模型任务准确率仍需使用可用的正式权重另行验证。

原始数据位于 [results/2026-11-17](../benchmark/qwen38/results/2026-11-17/)：

- `performance-{sglang,mini}.json`：各 24 组离线测量，含每次耗时与 min/max。
- `online-{sglang,mini}.json`：各 6 组 HTTP 测量，含逐请求 usage/耗时。
- `accuracy-*.json`：输入、参考 top-20、生成 token 和 teacher-forced 逐项误差。
- `tests-final.log`、`exclusive-final-session.log`、`model-manifest.json`：回归结果、停服/恢复证据和文件哈希。

复现入口为 `benchmark/qwen38/run.sh`，命令见 [使用说明](qwen38-next-local.md)。
