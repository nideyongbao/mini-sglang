# Qwen3.8 Flash Next：逐层精度定位与算子消融

> 后续严格默认模式、CUDA graph 精度及独占性能已补充在 [新报告](qwen38-strict-default-report.md)。本文中的默认路径和“尚未测量”均描述当时版本。

日期：2026-11-17。分支：`feat/qwen38-next-local`。本报告承接 `qwen38-next-local-report.md`，新增中间张量证据；此前源码审计中的“可能原因”不作为本轮结论。

## 实验边界

参考固定为本地镜像 `lmsysorg/sglang:dev-cu12-qwen38-next-local` 中的 SGLang（commit `4ccff141dbe992794f9da6c3aa23535b4f72000d`），H20 上使用已有 FA3 兼容补丁。checkpoint 是 `tiny-random/qwen3.8-flash-next-fp8`：8 层、hidden=32、6 GDN + 2 QSA、32 experts/top-10、block-FP8 32×32、HC=4/rank=4、带 PLE。这些结果不能替代正式训练权重的任务准确率。

双方 TP=1、相同 token IDs、batch、context=4096、prefill chunk=512、BF16 激活和 FP32 GDN 状态；逐层抓取使用 eager execution。主组输入长度 16/128/2048，batch=1。扩展组使用原有随机长输入 2048/3073，batch=1，以及随机输入 16/128，batch=2。

每次 forward 保存真实 logits 后再强制相同的后续 token；连续 decode 的输入历史一致，避免自由生成第一次分叉后继续比较不同计算。比较器逐帧断言输入和位置一致。参考可能多执行一个最终未返回的 decode，只比较双方共同的语义位置。主组 27 帧，扩展组 41 帧（57 行 logits）；chunk 中途的 logits 是诊断输出，不能全部当作用户得到的 next-token 样本。

抓取覆盖 embedding、每层 HC norm/mix/combine、GDN 输入及 core/输出、QSA QKV/归一化旋转后的 Q/K/core、indexer 投影及逻辑索引、路由 logits/专家/共享专家/合并、PLE 子模块、层输出、最终 HC 和完整 248320 维 logits。新增长 decode 的 index Q/K/score 捕获用于继续细分原因。不同物理页地址先还原为逻辑 token 位置。

## 已定位的分歧

| 场景与首个观察点 | 直接证据 | 参考路径与 mini 路径 | 消融结果 |
|---|---|---|---|
| 短 prefill，第 0 层 MoE 合并 | router、routed experts、未缩放 shared expert 在此之前一致；长度128合并最大差 `3.814697e-6` | 参考融合 gate GEMV + sigmoid + multiply + routed-add；原 mini 分开的 BF16 操作 | 对相同捕获输入，参考融合算子精确复现参考输出 |
| decode，第 0 层 GDN core | 该层 `mixed_qkv/a/b` 一致，core 开始分歧 | 参考 sigmoid(beta) 后转 BF16 再回 FP32；原 mini 保留 FP32 beta | 同一合成输入和初始状态下，补舍入后 output 与完整 FP32 state 均逐位一致 |
| 512-token prefill chunk，第 0 层 HC combine | mix/norm/attention 输出一致，combine 首先不同 | 本地 Triton FP32 tree reduction 实现 gate 投影，参考 BF16 linear/compiled combine | 相同输入 gate 投影有 767/2048 元素不同，最大差0.00390625；combine 最大差0.000244140625；开关 FMA 不消除差异 |
| 前三类替换后，第 3 层 QSA prefill | Q/K 准备一致，attention core 首先不同 | 原 mini 直接分页 sparse kernel；参考首 chunk sparse Triton、后续 chunk gather/cat + chunk sparse kernel | 换回参考 prefill 后主组完整 logits 精确一致帧从0提高到23/27 |
| 超过 QSA 预算后的 indexer | 投影一致，但 index Q/K 不同 | 参考 cos/sin 先 BF16，每个乘法和加法各自舍入；原 mini 使用 FP32 旋转后一次 BF16 | 改成参考舍入后，检查的第3层 decode index Q、压缩K、有效scores均逐位一致 |
| indexer 有效 scores 一致仍选不同块 | 同一个参考 fast_topk、相同输入，连续调用得到不同并列集合 | 非确定性并列 top-k；形状也影响 kernel 分派，但固定形状仍发生 | 513选512时39个候选同为最低0分，8次相同宽度528的调用出现多种剔除块；不能只靠“调用相同算子”保证同一稀疏子图 |

GDN 状态逐位一致的单算子证据使用固定种子的相同合成初态，不等于所有模型层运行状态均已逐位对齐。整网中，上游 QSA prefill 的不同结果会写入后续 GDN 的状态，因此换回 decode 核后，第4层仍可能继承 prefill 误差。

HC 的实验排除了“只要启用 FMA 就一致”的解释。它证明当前尺寸下 gate 投影的归约路径有差异，但未把所有 HC 编译子算子的误差拆成独立贡献。

进一步对随机 batch=2 残余分歧定位：稳定QSA top-k后，128-token packed prefill 最先在第2层 routed experts 出现 `1.1920929e-7` 差异（8192个元素中1个），并传播到后续压缩K状态。相同输入下，两边 MoE expert IDs 一致，但路由权重最大差 `2.2351742e-8`。原 mini 的 AOT topk-softmax 后还做二次归一化；只删去二次归一化仍不逐位一致。整体换为参考 `fused_topk` 分派后，扩展组完整logits达成41/41帧逐位一致。因此“相同FP8 GEMM”并不意味着它收到的路由权重完全相同。

## 额外发现：短 batch decode 的 NaN

随机长度16、batch=2 的原路径，在第3层 QSA decode core 首次产生 NaN；参考输入和对应输出有限。共享/HC/GDN 等替换不能解决。对 FA3 gather 临时 K/V 缓冲区清零后，同一用例全部输出有限。进一步固定相同有效Q/K/V，FA3临时缓冲尾部填0时768个输出均有限，尾部填NaN时768个输出全部NaN；这排除了模型上游作为该用例的必要原因。

本轮性能所有 mini 候选统一使用这一稳定性修正，原始非有限结果保留为失败证据，不参加有限误差平均。

该问题说明，仅检查返回 token 数量不够：NaN logits 仍可能被 argmax 转成某个 token。性能脚本现在在计时外检查实际采样 logits / 参考输出 logprob 有限，然后才预热与计时。

## 精度结果与局限

所有普通top-k候选最终都重新与同一份已冻结的参考capture比较，避免参考重跑的非确定性混入消融表。

| 路径 | 完整logits一致帧 | 严格top-1行 | 全词表RMS | 全词表最大绝对差 |
|---|---:|---:|---:|---:|
| 主组：旧路径 | 0/27 | 27/27 | 0.000268908 | 0.00195312 |
| 主组：默认修正 | 0/27 | 27/27 | 0.000216388 | 0.00195312 |
| 主组：参考HC/GDN/shared | 0/27 | 27/27 | 0.000211912 | 0.00195312 |
| 主组：再替换参考QSA prefill | 23/27 | 27/27 | 8.1911e-05 | 0.0010376 |
| 主组：参考组合，普通top-k | 23/27 | 27/27 | 8.74693e-05 | 0.000976562 |
| 主组：参考组合+router，双方稳定top-k | 27/27 | 27/27 | 0 | 0 |
| 扩展组：默认修正 | 0/41 | 55/57 | 0.000236856 | 0.00195312 |
| 扩展组：参考组合，普通top-k | 22/41 | 57/57 | 0.000109437 | 0.0012207 |
| 扩展组：参考组合，双方稳定top-k | 36/41 | 57/57 | 3.41034e-05 | 0.000976562 |
| 扩展组：再换参考router，双方稳定top-k | 41/41 | 57/57 | 0 | 0 |

默认修正在主组把RMS降低约 **19.5%**，但最大误差仍为0.001953125，完整logits一致帧仍为0/27。扩展组严格top-1为55/57，参考最高值并列集合命中56/57；其中一个decode位置仍有0.0009765625的非并列间隔，不能把所有剩余差异解释成并列选择。旧路径扩展组有7个非有限帧，有限的34帧/43行之外不能当作通过。

最终严格诊断组合在这7组固定输入/batch场景中，**68/68帧、84/84行完整logits逐位一致，最大误差和RMS均为0**。84行包含chunk中途诊断输出。这一结论需要参考HC/GDN/QSA/归约/router路径以及双方相同的稳定QSA top-k；它不是默认快速路径的结果，不代表任意输入、正式权重、CUDA graph或部署并发下已经普遍逐位一致。最后加入router和稳定top-k的严格组合没有单独做独占性能测量，不能套用下面快速路径吞吐。


“完整 logits 一致帧”是逐位比较，不是仅比较 top-1。不同替换的最大绝对误差未必单调下降，因为 BF16 舍入、top-k 并列和状态传播会改变误差位置。不能把单项替换的改善线性相加。

正常参考的 fast_topk 非确定性是本轮额外识别的限制。要检查整个数值路径是否等价，必须另外做双方同样的确定性并列处理实验；这会修改参考的选择策略，应单独标注，不能冒充未修改参考的性能或精度。

## 独占 GPU 性能消融

输出128 tokens的中位延迟（ms），四列分别为输入长度/batch：

| 路径 | 128/1 | 128/4 | 2048/1 | 2048/4 |
|---|---:|---:|---:|---:|
| 原算术 + scratch清零 | 130.43 | 139.28 | 185.34 | 360.73 |
| 仅参考shared merge | 124.34 | 136.48 | 179.42 | 352.46 |
| 仅beta舍入 | 129.10 | 140.37 | 183.25 | 352.61 |
| 仅参考HC | 174.82 | 197.21 | 250.17 | 515.10 |
| 仅参考QSA prefill | 130.39 | 142.74 | 189.22 | 371.52 |
| 本轮默认修正路径 | 122.50 | 135.44 | 177.54 | 353.79 |
| HC/GDN/QSA组合替换（普通top-k） | 164.70 | 186.54 | 241.28 | 513.23 |
| SGLang参考 | 172.53 | 177.79 | 257.87 | 597.67 |

本轮默认修正路径的离线吞吐分别为 **1044.9 / 3780.3 / 721.0 / 1447.2 tokens/s**（以原始JSON计算值为准），对应参考的 **1.41× / 1.31× / 1.45× / 1.69×**。相对于统一清零后的旧算术路径，延迟下降约1.9%–6.1%；其中小幅变化需要更多重复才能作为稳定收益承诺。

单独换回参考HC使输出128的端到端延迟增加 **34.0%–42.8%**；这是tiny HC融合确实贡献性能优势的直接消融证据。单独换回QSA prefill在这四组输出128任务中的变化约 **−0.03%至+2.99%**，不能据此声称QSA解释了全部加速。

输出1 token的中位延迟（ms）：

| 路径 | 128/1 | 128/4 | 2048/1 | 2048/4 |
|---|---:|---:|---:|---:|
| 原算术 + scratch清零 | 13.93 | 18.44 | 54.53 | 220.69 |
| 仅参考shared merge | 13.92 | 18.78 | 54.68 | 219.33 |
| 仅beta舍入 | 13.51 | 17.85 | 53.42 | 213.87 |
| 仅参考HC | 20.16 | 24.74 | 80.72 | 326.47 |
| 仅参考QSA prefill | 14.73 | 20.06 | 58.96 | 235.60 |
| 本轮默认修正路径 | 13.90 | 18.49 | 55.36 | 222.70 |
| HC/GDN/QSA组合替换（普通top-k） | 20.47 | 24.92 | 83.29 | 334.32 |
| SGLang参考 | 29.36 | 30.94 | 115.77 | 454.20 |


默认修正列是在合入默认实现前，通过等价的算子替换组合测得；合入后另做功能回归。

所有测量关闭 tensor hooks 和强制 token。两边 CUDA graph 捕获 batch1–8，context4096、chunk512、输入长度128/2048、batch1/4、输出1/128，逐形状检查有限输出，预热2次、测量5次取中位数。输出1的延迟包含 prefill、框架和调度开销，不是纯 CUDA attention 时延；输出128的吞吐同样是离线 generate 端到端值。这里没有重跑 HTTP、流式 TTFT/ITL 或并发到达率扫描。

算子替换也可能增加 gather、Python 分派与内存分配；表中差值是“这条可运行替换路径的总代价”，不是某个 kernel 自身的耗时。此前 1.17–2.05× / 1.33–2.31× 是原离线/HTTP矩阵，本轮应使用这里的独立消融结果解释，不能沿用为修正后的性能承诺。

## 后续部署原生正式权重的路径

1. **固定原生 checkpoint 与兼容性清单。** 核对 `Qwen4ExpForConditionalGeneration/qwen4_exp_text`、层类型顺序、HC/PLE/QSA 参数、ngram hash/分片、RoPE/MRoPE、共享专家、量化模块白名单、scale含义和tokenizer/template。当前本地 `Qwen3.8-27B-FP8` 的 config 是 `qwen3_5`，不能用来证明 Flash Next 原生正式权重已支持。
2. **先建立参考部署模式。** 在正式权重上先固定 TP=1、text-only、无 prefix reuse、无推测解码，相同 GDN 状态精度、FP8块大小/激活量化、QSA预算和 prefill/decode 后端。参考模式以可比性为先，明确并列 top-k 是允许集合差异，还是双方启用稳定按索引决胜。稳定策略应按(score, block_id)作字典序，不能只固定随机种子或任意加epsilon改变原分数。当前替换脚本是诊断 harness，不能作为正式服务开关直接上线。
3. **先完成加载与内存工程。** 目前 loader 一次把 safetensors 放到 GPU 并做拼接，正式专家权重与 PLE 表会有高峰值；需要按层/分片流式加载、直接写目标张量、处理FP8专家格式与scale，PLE避免常驻整表BF16副本。核算权重、PLE、GDN状态、普通KV、QSA压缩/待完成组、graph scratch和并发请求的总内存。不能仅扩大 `max_seq_len` 或替换模型路径。
4. **以正式权重分阶段验收。** 先短 prefill → 长 chunk prefill → teacher-forced decode → batch/槽位重用与取消 → graph/padding → 超预算稀疏选择。比较全词表有限性、logits RMS/最大差、分布/KL、非并列top-1、GDN/PLE/QSA状态、路由与selected-block集合，最后跑真实任务准确率/困惑度。阈值由该模型基线重复性确定，不能把 tiny 的0.005误差容限直接搬过去。
5. **逐个放开优化与规模。** 对真实 hidden/expert 尺寸单独选择 HC/MoE/QSA kernel，并复用本轮逐层与消融流程。TP/EP需补齐HC、GDN、PLE、indexer状态和通信切分；前缀缓存必须联合恢复GDN+PLE+QSA状态。MTP、speculative、视觉、多卡均应作为独立支持项，不从当前 TP=1 tiny 结果推断。
6. **服务上线按实际负载验收。** 固定请求到达率与长度分布、报告吞吐/TTFT/ITL/p95/p99和显存峰值；预填充与解码分别profile。测试独占GPU与共享GPU条件，保存镜像/权重/配置/代码哈希；再灰度上线并保留参考服务回退。小模型的固定开销优势不会按比例迁移到大模型。

## 可复现材料

脚本见 `benchmark/qwen38/alignment/README.md`。结构化误差、耗时、日志和manifest位于 `benchmark/qwen38/results/2026-11-17-alignment/`。较大的原始 `.pt` 张量存放在仓库外的 `../qwen38-alignment-artifacts/2026-11-17/`，不把GB级文件提交Git。

本轮回归：16项GPU测试通过（含beta输出/状态逐位比较和短batch、NaN未使用缓存测试）；合入默认实现后，batch=2、输入16/2048、输出1/128的CUDA graph有限性检查通过。该graph smoke与原服务共存，仅作功能验证，不进入独占性能表。

原服务于 **2026-11-17 13:36:46 +0800** 恢复健康；镜像、启动命令、入口、环境变量映射和端口保持一致（环境变量列表顺序不同），`/v1/models`仍为`qwen`。性能期间的开始和结束GPU显存均为0 MiB，现有服务停止后再串行测量。

保存了971个原始文件，约6.08 GB，逐文件SHA-256见artifact manifest；Git内仅提交约1.1 MB的结构化摘要、日志和复现代码。
