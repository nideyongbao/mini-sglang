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

## 限制与结果文件

支持范围为目标本地 tiny checkpoint 的 TP=1、纯文本、BF16 激活、block-FP8 路由专家。未验证 4096 以上上下文、其他 GPU、真实大模型 checkpoint、MTP/推测解码、视觉和多 GPU；不支持前缀 PLE/QSA 状态复用，强制 naive cache。PLE embedding 查表保存 BF16 副本并在 lookup 时应用原 scale，专家 GEMM 保留 FP8。

性能优势不能外推到 27B/更大模型或不同后端；这里只有 5 次重复，没有在线流式 TTFT/ITL/p99 或真实请求到达率曲线。全词表数值指标和真实模型任务准确率仍需使用可用的正式权重另行验证。

原始数据位于 [results/2026-11-17](../benchmark/qwen38/results/2026-11-17/)：

- `performance-{sglang,mini}.json`：各 24 组离线测量，含每次耗时与 min/max。
- `online-{sglang,mini}.json`：各 6 组 HTTP 测量，含逐请求 usage/耗时。
- `accuracy-*.json`：输入、参考 top-20、生成 token 和 teacher-forced 逐项误差。
- `tests-final.log`、`exclusive-final-session.log`、`model-manifest.json`：回归结果、停服/恢复证据和文件哈希。

复现入口为 `benchmark/qwen38/run.sh`，命令见 [使用说明](qwen38-next-local.md)。
