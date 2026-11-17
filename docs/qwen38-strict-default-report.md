# Qwen38 strict 默认模式：CUDA graph 精度与独占性能

> 后续补充：[相同输入、chunk512、单并发离线/HTTP复测](qwen38-single-concurrency-report.md)。

本轮已达到：默认strict、CUDA graph下128步主矩阵4992/4992完整logits逐位一致，fast可选；严格模式相对fast延迟增加27.6%–45.4%。原服务已恢复，strict测试服务在本机31039可用。逐位一致参考采用双方相同的确定性QSA top-k，原top-k SGLang另列性能基线。

日期 2026-11-17，分支 `feat/qwen38-next-local`。本轮把此前诊断替换组合迁为正式模型配置；API/CLI 默认 `strict`，显式 `fast` 才启用近似快速核。

## 比较口径

固定 H20、镜像 `lmsysorg/sglang:dev-cu12-qwen38-next-local`、镜像内 SGLang commit `4ccff141dbe992794f9da6c3aa23535b4f72000d`，同一 tiny-random FP8 checkpoint。TP=1、BF16 激活、FP32 recurrent state、context4096、token pool32768、max running8、FP8 GEMM/MoE Triton。三方均开启 CUDA graph，捕获 batch1–8，无 padding；额外测严格 eager 消融。

需要区分两条 SGLang 基线：

- **SGLang deterministic**：H20 FA3 兼容修正 + 稳定 QSA top-k（同分按 block ID 决胜）。这是逐位对齐基线，所有精度表和 HTTP 表以此为参考。
- **SGLang original-topk**：保留 H20 FA3 修正，使用镜像原 top-k。该算子在并列候选上非确定；本报告另测其离线性能，不能承诺 strict 与它每次选择相同稀疏子图。

固定 batch 离线测试输入长度128/2048/3072，batch1/4/8，输出1/128；双方 prefill budget 和 SGLang max_prefill_tokens 均为32768，以一次完整 prefill 接纳整个 batch。输入由固定种子生成，JSON 记录 SHA-256；解码统一 teacher-force token1000，双方采样后各做一次 GPU fill，所有后续 GDN/PLE/QSA 历史一致。计时区没有中间张量抓取；每个形状先校验有限分布、再预热，5次取中位数，保留每次耗时。输出1是 generate 首 token 端到端延迟；输出128是整体吞吐，均不是纯 kernel 时间。

HTTP 使用 chunk512、同一 chat 输入、并发1/4/8、输出128、自然生成、2次预热+5次测量。两个调度器的实际内部 batch 可能不同，因此 HTTP 是服务对比，不能冒充逐帧相同执行。此前 chunk512 的离线数据也不能直接套用到本轮 full-prefill 固定 batch 矩阵。

## 为什么又需要两处修正

扩大至 batch8 和长输入后，原严格组合并非普遍逐位一致：

1. 长度16、batch8，最先在第1层 GDN 输出门控 norm 分歧。core 一致，本地 `gated_rmsnorm_` 与参考 FLA `layernorm_fn(..., activation="sigmoid")` 在边界值舍入不同。换参考 norm 后，16步测试从237/256帧提高到253/256帧。
2. 长度3073、batch4，第3层 QSA index投影一致，但4个 prefill query 的选择索引不同，并污染后续压缩状态。严格模式改为参考 `qsa_index_q_norm_rope_store` 与 `qsa_index_k_compress_store`，取代分开的 FlashInfer norm、PyTorch pooling/RoPE 准备。随后256/256帧一致，并通过下方128步扩展测试。

这也说明“短输入 top-1 一致”不足以验收严格路径。旧失败摘要和逐层张量保留在归档中。

## 正式两种模式的路径

| 模块 | strict（默认） | fast（可选） |
|---|---|---|
| HC mix/combine | SGLang GatedResidual + 参考 linear/compiled 算术 | tiny Triton 融合 HC |
| MoE router | SGLang fused_topk 分派（本镜像 JIT） | 本地 AOT topk-softmax/归一化 |
| FP8 experts | 相同 block-FP8 Triton GEMM；参考 activation/reduction | FP8 GEMM保留，使用本地 activation/reduction 路径 |
| shared expert merge | 参考 fused_gate_sigmoid_mul_add | 同一参考融合算子 |
| GDN prefill / decode | 参考 prefill；参考 packed recurrent decode | 同一 prefill；本地 packed decode，保留 BF16 beta 舍入修正 |
| GDN 输出门控 norm | 参考 FLA layernorm_fn | 本地 gated_rmsnorm_ |
| QSA index Q/K准备 | 参考融合 Q norm/RoPE store、K compress store | 分开的 norm、pooling、BF16分步RoPE |
| QSA score / selection | TileLang MQA + stable score/ID top-k | 本地 score计算 + 原 fast-topk |
| QSA prefill | 参考 sparse Triton / chunk sparse Triton | 本地分页 sparse attention |
| QSA decode | FA3，未使用 scratch 清零 | 同一FA3及清零修正 |
| Embedding / PLE / 页表 | 保留本地状态管理、查表及卷积；由整网逐位测试覆盖 | 同一实现 |
| CUDA graph | 精确 batch replay；未捕获大小走 eager | 原 graph replay/padding策略 |

这里“严格对齐”指数值关键算子、舍入、选择策略及实际模型 batch 对齐。Mini-SGLang 仍负责自己的调度、页表、PLE/GDN/QSA 状态存储和 gather；没有把请求转发给 SGLang Engine，也不声称两个框架每一次内存搬运和 kernel launch 序列完全相同。


## CUDA graph 精度结果

完整248320维logits在强制相同后续输入前保存，按比特表示逐元素比较，不只检查top-1。比较器要求所有输出帧存在，输入ID、decode位置和logits形状一致。SGLang overlap可能额外执行一个未返回decode，精度比较排除它；端到端计时保留该运行时开销。

| 组别 | strict 完整向量逐位一致 | fast top-1一致 | fast 完整向量逐位一致 |
|---|---:|---:|---:|
| 与性能矩阵相同的9组输入、128步、固定token1000 | 4992/4992（100%） | 4764/4992（95.43%） | 0/4992 |
| 长度16/128/2048/3073、batch1/2/4/8、16步变化续接 | 960/960（100%） | 949/960（98.85%） | 0/960 |
| chunk512，batch1，长度16/128/2048/3073，16步 | 64/64（100%） | 未重测 | — |
| graph边界：长度1/3/4/16/2049，batch3/5/6/7，16步 | 1680/1680（100%） | 未重测 | — |

strict各组最大误差与RMS均为0。128步组fast最大误差 **0.0029296875**，RMS **0.00028978669**；参考最高值并列集合命中4851/4992，其中141个位置不属于参考并列最优集合，不能把全部top-1差异归因于同分选择。

这些百分比是tiny随机权重的**数值一致率**，不是正式模型的语义任务准确率。结论限定本镜像、权重、输入、形状和参数。HTTP动态调度可能改变实际forward batch，不能据此承诺任意并发服务与另一套调度器逐token复现。

## 独占离线性能：CUDA graph开启

输出128 tokens，单位ms；吞吐比值越高代表Mini越快。“确定性/原版”分别指上文两条SGLang基线。

| 输入/batch | SG确定性 | SG原top-k | mini strict | mini fast | strict/SG确定性 吞吐 | fast/SG确定性 吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| 128/1 | 174.70 | 179.41 | 168.53 | 121.72 | 1.037× | 1.435× |
| 128/4 | 179.91 | 183.59 | 193.16 | 132.84 | 0.931× | 1.354× |
| 128/8 | 181.50 | 181.73 | 208.71 | 150.34 | 0.870× | 1.207× |
| 2048/1 | 177.81 | 181.06 | 181.15 | 135.43 | 0.982× | 1.313× |
| 2048/4 | 185.38 | 184.47 | 213.01 | 150.66 | 0.870× | 1.230× |
| 2048/8 | 198.59 | 183.90 | 225.92 | 170.65 | 0.879× | 1.164× |
| 3072/1 | 178.63 | 179.06 | 184.25 | 135.73 | 0.969× | 1.316× |
| 3072/4 | 190.31 | 183.98 | 213.94 | 152.41 | 0.890× | 1.249× |
| 3072/8 | 235.71 | 183.86 | 229.69 | 180.02 | 1.026× | 1.309× |

strict相对fast延迟增加 **27.6%–45.4%**，等价吞吐降低 **21.6%–31.2%**。与确定性SGLang相比，strict吞吐 **0.870–1.037×**、fast **1.164–1.435×**；与原top-k SGLang相比，strict **0.800–1.065×**、fast **1.021–1.474×**。

稳定选择也有代价：3072/batch8时，SGLang确定性版本235.71ms，原top-k版本183.86ms，延迟增加 **28.2%**。短输入几%的反向差值处于本次串行测量波动范围，不应解释为稳定排序一定更快。

输出1 token端到端延迟（prefill及框架开销），单位ms：

| 输入/batch | SG确定性 | SG原top-k | strict | fast |
|---|---:|---:|---:|---:|
| 128/1 | 30.16 | 39.61 | 21.65 | 13.23 |
| 128/4 | 31.58 | 31.41 | 28.28 | 17.76 |
| 128/8 | 32.48 | 32.32 | 36.12 | 23.14 |
| 2048/1 | 30.90 | 30.68 | 22.75 | 13.65 |
| 2048/4 | 32.03 | 32.12 | 28.45 | 19.47 |
| 2048/8 | 36.48 | 34.03 | 36.76 | 28.04 |
| 3072/1 | 31.23 | 30.63 | 23.52 | 13.77 |
| 3072/4 | 33.16 | 32.45 | 29.95 | 21.48 |
| 3072/8 | 74.10 | 35.89 | 39.05 | 36.68 |

## 保持精度的优化与CUDA graph消融

| 输入/batch | strict graph ms | strict eager ms | graph加速 | 保留本地GDN ms |
|---|---:|---:|---:|---:|
| 128/1 | 168.53 | 2259.93 | 13.41× | 166.87 |
| 128/4 | 193.16 | 2374.75 | 12.29× | 193.30 |
| 128/8 | 208.71 | 2404.53 | 11.52× | 206.65 |
| 2048/1 | 181.15 | 2239.04 | 12.36× | 181.14 |
| 2048/4 | 213.01 | 2379.48 | 11.17× | 212.22 |
| 2048/8 | 225.92 | 2397.97 | 10.61× | 224.11 |
| 3072/1 | 184.25 | 2237.54 | 12.14× | 182.11 |
| 3072/4 | 213.94 | 2387.03 | 11.16× | 213.16 |
| 3072/8 | 229.69 | 2389.42 | 10.40× | 229.05 |

- **保留精确batch CUDA graph。** 本轮提升10.40–13.41倍，并通过graph replay数值测试。未捕获的batch回退eager；服务显式捕获1–8。这里eager只关闭decode graph，参考HC的torch.compile仍启用。
- **保留安全的数据搬运和状态优化。** 原生调度、页表、整数ngram hash、窄embedding/KV store、FA3 scratch清零、已对齐的shared-expert融合及beta舍入修正已进入整网验证。动态合批本身不保证跨框架同一batch。
- **本地GDN recurrence是候选，暂不默认开启。** 在960个完整logits向量上仍逐位一致，另有输出/FP32 state逐位单算子测试。端到端延迟变化约−1.16%至+0.07%，接近噪声。strict默认继续用参考核；消融开关仅存在于benchmark。
- **HC快速融合、本地门控norm、非确定top-k留在fast。** 这些会改变舍入或稀疏子图。此前HC单项消融已证明其性能贡献；本轮27.6%–45.4%是整套模式差价，不能全部归到一个算子。
- **下一步优化确定性top-k与缓存准备。** 当前stable argsort是保守实现，全量排序及无效列带来开销。可做保持(score, block ID)顺序的确定性选择核、预分配scratch、减少QSA索引/页表重建和gather复制；尚未实现或计入本报告，合入前需逐位与性能复验。

## HTTP性能与可用性

参考为确定性SGLang，chunk512，三方graph batch1–8；输出128。单位ms为一批HTTP请求完成耗时，吞吐按并发×128/耗时计算。

| 实际prompt tokens/并发 | SG ms | strict ms | fast ms | strict/SG吞吐 | fast/SG吞吐 |
|---|---:|---:|---:|---:|---:|
| 140/1 | 178.35 | 169.48 | 124.78 | 1.052× | 1.429× |
| 140/4 | 243.74 | 218.56 | 150.52 | 1.115× | 1.619× |
| 140/8 | 523.00 | 257.79 | 184.73 | 2.029× | 2.831× |
| 2060/1 | 309.56 | 274.32 | 194.44 | 1.128× | 1.592× |
| 2060/4 | 669.67 | 555.80 | 373.16 | 1.205× | 1.795× |
| 2060/8 | 1315.26 | 913.19 | 609.24 | 1.440× | 2.159× |

HTTP strict吞吐为参考 **1.052–2.029×**，fast为 **1.429–2.831×**。严格模式在固定batch离线未必更快，HTTP仍可受益于调度与chunk执行开销；不能据此反推严格算子本身快2倍。没有测原top-k SGLang的HTTP矩阵，也没有进行到达率或p95/p99容量扫描。

三条服务均通过流式/非流式文本一致、断流后继续处理请求、两轮8并发混合输出长度、槽位重用和重复输出检查。相同140-token prompt、16-token自然生成时，strict与SGLang文本相同（API可选空字段不同）。23项回归测试通过。

## 默认部署与正式模型迁移

本机另启默认strict测试服务：容器 minisgl-qwen38-strict，地址 http://127.0.0.1:31039/v1。使用原始镜像、新容器和仓库正式实现，没有安装诊断替换器。它服务tiny-random模型，输出没有语义用途。原sglang-qwen38-dflash2于 **15:02:37 +0800** 恢复健康；镜像、命令、环境变量映射、挂载映射、端口一致，模型名仍为qwen。实验容器停止，保留原服务和strict测试服务；共享GPU后的功能检查不计入独占性能。

从仓库启动（先停止同端口服务，或选择其他PORT）：

~~~bash
# 默认strict
bash benchmark/qwen38/run.sh serve

# 可选fast
QWEN38_MODE=fast PORT=31040 bash benchmark/qwen38/run.sh serve
~~~

API为 LLM(model_path, qwen38_mode="strict", ...)，默认即strict；CLI通过 --qwen38-mode fast切换。--cuda-graph-bs 1 2 3 4 5 6 7 8指定完整batch集。模式在构造时固定，不在正在服务的模型实例上热切换。

正式原生Qwen3.8 Flash Next需固定真实checkpoint/tokenizer/template、HC/PLE/QSA参数、量化block/scale和算子版本。沿用strict作为验收起点，不能只替换model路径就认定完成。当前限定TP=1、纯文本、BF16激活、FP8专家、naive cache，无视觉/MTP/推测解码/前缀状态复用。正式尺寸先补按层/分片加载及PLE表内存管理，避免一次加载/拼接与BF16表副本的内存峰值。

真实权重需重跑短/长prefill、跨chunk状态、teacher-forced decode、部署batch及graph形状、EOS/取消/槽位复用；再测困惑度和任务准确率。固定GPU、CUDA/torch/SG版本、TP、量化、context及确定性选择作为复现契约。增加TP/EP、prefix复用或MTP时分别实现状态与通信语义并重新验收。tiny吞吐倍数和数值误差不能外推到大模型。

## 材料与复现

复现入口见 [modes/README](../benchmark/qwen38/modes/README.md)。逐帧误差、所有计时样本、服务检查、日志、恢复校验见 [results/2026-11-17-modes](../benchmark/qwen38/results/2026-11-17-modes/)。exclusive.sh记录本机已授权停服/恢复过程；常规run.sh不会自动停止业务服务。

原始tensor在仓库外 ../qwen38-alignment-artifacts/2026-11-17-modes/，raw-manifest.json记录逐文件SHA-256，environment.json记录环境、checkpoint与源码指纹。计时开始前GPU显存0MiB、利用率0%；结束时761MiB来自最后一个测试服务，随后停止并恢复原服务。原服务无请求也占显存，有请求还会争用计算/带宽，本轮所有计时均在停服后串行完成。
