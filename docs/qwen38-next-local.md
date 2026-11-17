# Qwen3.8 Flash Next 本地适配

本分支为 `Qwen4ExpForConditionalGeneration` 的纯文本推理增加原生 Mini-SGLang 模型实现，目标权重为 `/shard_data/brooksli/models/tiny-random/qwen3.8-flash-next-fp8`。调度、请求状态、KV 分页、CUDA Graph 和 HTTP 服务由 Mini-SGLang 管理；复用镜像内 SGLang 的底层 FP8 MoE、QSA top-k 算子，不通过 SGLang Engine 转发请求。

实测结果和限制见 [精度与性能报告](qwen38-next-local-report.md)。

## 环境与启动

使用本机已有镜像 `lmsysorg/sglang:dev-cu12-qwen38-next-local`。这份实验依赖镜像中的私有 Qwen4Exp 配置和底层算子 API；普通 PyPI 依赖安装不等价于此环境。

在仓库根目录执行：

```bash
# 启动原生 Mini-SGLang HTTP 服务，默认端口 31039
bash benchmark/qwen38/run.sh serve

# GPU 算子回归
bash benchmark/qwen38/run.sh test

# 启动参考服务并执行短/长上下文精度对比
bash benchmark/qwen38/run.sh accuracy

# 确保 GPU 上没有其他推理任务后，串行运行两个后端
bash benchmark/qwen38/run.sh performance

# 原生 HTTP E2E 测量，同样串行执行且独占 GPU
bash benchmark/qwen38/run.sh online-sglang
bash benchmark/qwen38/run.sh online-mini
```

脚本支持 `MODEL_PATH`、`IMAGE`、`RESULTS_PATH` 和服务模式的 `PORT` 环境变量；默认结果目录 `benchmark/qwen38/results/local`。脚本不会自动停止已有业务服务。首次运行需要 JIT 编译，编译和模型加载不计入稳态性能。

```bash
curl http://127.0.0.1:31039/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/model","messages":[{"role":"user","content":"Hello"}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 实现范围

- TP=1、BF16 激活、block-FP8 routed experts、BF16 shared expert。
- GDN 使用该架构的 sigmoid 输出门，支持 head_dim=32 的 RoPE、embedding 和 KV 存储。
- HyperConnection 分组 Gemma RMSNorm、低秩输入混合和残差注入；目标 tiny 配置提供融合 Triton 实现。
- PLE 数字顺序拼接 FP8 embedding 分片，保留 int64 哈希参数、EOS 边界、跨 chunk 的 n-gram 和卷积状态。
- QSA 压缩 key、因果块选择与尾部 token，预填充采用稀疏 attention，解码打包选中的 KV 后调用 FA3。
- 请求槽位复用时重置 PLE/QSA 历史；支持跨压缩块边界的 chunked prefill 和 CUDA Graph 解码。

未指定上下文上限时默认取 QSA budget（该模型为 2048），此范围内 QSA 与全量因果注意力等价。部署和性能脚本显式指定 4096，以覆盖真实稀疏选择。4096 以上上下文尚未验证。

本地适配不支持多模态、TP>1、dummy weights 或前缀状态复用。加载时跳过视觉和 MTP 参数，强制使用 naive cache；不启用 MTP/推测解码。FP8 PLE embedding 在加载时转为 BF16 查表，并在 lookup 时应用原 scale；路由专家保留实际 FP8 GEMM，未整体反量化成 BF16 模型。

## 参考端兼容修正

`patch_reference.py` 仅修改测试容器内 QSA packed-attention 的 FA resolver，优先使用已经安装的 `sgl_kernel` FA3，避开当前镜像 FA4/CUTLASS 在 H20 上的接口异常。原文件保存在相邻 `.py.original`。不修改宿主镜像和已部署的 DFlash2 服务。

参考服务使用 `--language-only`，并捕获 batch 1–8 的全部 graph、关闭 graph padding；本机镜像在部分非精确 batch 的 padding 路径曾返回非有限 logprob。基线因此是经过这项兼容修正且显式配置的 SGLang，不代表镜像无参数即可运行。

## 精度解释

输入 token、采样方式、checkpoint 固定。精度脚本既比较自由生成，也沿 SGLang 输出进行 teacher forcing，在相同历史下对比参考 top-20 token 的 logprob。默认绝对误差容限为 0.005，可通过 `--max-logprob-error` 修改。这是本实验的数值回归阈值，不是模型官方认证标准。

随机模型中经常存在 BF16 概率并列：不同实现选择不同并列 token 后，后续自由生成会分叉。因此报告同时保留严格 top-1、参考并列集合命中率、自由生成一致率和 logprob 误差，不用单一指标宣称完全一致。该随机权重不能用于判断真实模型的语义质量。
