<!-- LANG -->
[English](README.md) | **中文**

# 在 DGX Spark (GB10) 上运行 LongCat-Next —— 全模态推理服务

在单台 **NVIDIA GB10 系统（`sm_121`）** 上，通过**单个 SGLang 进程**运行[美团 **LongCat-Next**](https://huggingface.co/meituan-longcat)（75B 总参 / ~A3B 激活的任意到任意多模态 MoE：LongCat-Flash 主干 + MLA 注意力 + N-gram 过嵌入，视觉与音频均为原生 RVQ 分词器），并提供**兼容 OpenAI 的全模态接口**：

| 能力 | OpenAI 接口 | 状态 |
|---|---|:--:|
| 文本生成（支持 `stream`） | `POST /v1/chat/completions` | ✅ |
| 图像 / 音频 / 视频**理解** | `POST /v1/chat/completions` | ✅ |
| **图像生成**（文本 → 图像） | `POST /v1/images/generations` | ✅ |
| **声音克隆 TTS**（多句、支持 `stream`） | `POST /v1/audio/speech` | ✅ |
| 工具 / 函数调用 | `POST /v1/chat/completions`（`tools`） | ✅ |
| Anthropic Messages API（Claude Code） | `POST /v1/messages` | ✅ |

<sub>（LongCat-Next 不支持视频**生成**，视频仅支持理解。）</sub>

量化为 **`w8a8_int8`**（8-bit 权重 + 逐 token int8 激活）—— 切换到 8-bit 正是图像与音频生成变得连贯的关键（4-bit 下两者都崩溃）。单个自包含约 90 GB 模型，在单台 GB10 上稳定运行，并通过 [7/7 自检](#自检)端到端验证。下载权重之前，可先看 `examples/` 中的示例生成图像与语音片段。

> **构建历程** —— 调试与优化的完整过程（两个看似同一症状的独立问题、一次对抗式多智能体评审、一场推翻了自己两条既有结论的内核启动延迟排查，以及下文每一个性能数字对应的实测记录）都在
> **[research/FINDINGS.md](research/FINDINGS.md)** 中。本 README 中凡带数字的结论均为实测，而非估算。

> 专为 GB10 超级芯片（`sm_121`）构建 —— 已在 DGX Spark 上验证，预期可在任何基于 GB10 的机器上运行
> （依赖的是芯片而非具体产品）。只有 cu130 的 SGLang 基础镜像（`v0.5.16-cu130`）能为 `sm_121` 编译 Triton；在其他 GPU 上不保证运行。

## 性能（GB10 实测，未注明处均为单请求）

| 路径 | 首次发布时 | 当前默认 | 手段 |
|---|---|---|---|
| 单张图像，热态（1040×1040 + 精修） | 约 4–5 分钟 | **约 2.4 分钟** | 精修器默认关闭引导项、生成头逐层 CUDA 图重放、稠密 SDPA 头注意力 |
| TTS 生成速率 | 比实时慢约 2.2–2.4 倍 | **约 1.4 倍** | 音频头 FFN int8（每帧 −34–36 %） |
| TTS 首段音频（流式） | 约 32 秒（整段完成后） | **约 6.5 秒** | 滑窗分块声码 + PCM 流式输出 |
| 文本解码 | — | +6 %（单流）、**16 并发聚合 +13.6 %** | CUDA 图默认开启，max-bs 32 |
| 热态 agent 前缀（15.6k token） | 约 5.9 秒 | **约 0.36 秒** | 前缀（radix）缓存默认开启 |

以下各项**默认开启**，因为实测更快且无质量代价：CUDA 图（`LCN_CUDAGRAPH=1`）、生成头逐层图重放（`LCN_HEAD_GRAPH=1`，捕获时逐位相等校验，数学上与 eager 完全一致）、稠密 SDPA 头注意力（逐位一致）、前缀缓存、精修器关闭引导项（`LCN_REFINER_CFG_RANGE=1.0,0.0`，每图 −17 %，输出经人工判定无差异）。

以下默认项带有一个**已经人工判定的取舍**：`LCN_INT8_HEADS=audio` —— 音频生成头运行 int8（即上表 TTS 提速；听测未发现差异）。**视觉**头刻意保持 bf16：int8 在 5 对 5 同提示词对比中可测地增加了空间几何错误。设 `both` 可恢复全 int8（图像再快约 8 %），代价自负。代码中还保留了一个 int4 音频变体，**听测被否决** —— 已作为死路记录在案。

已知上限：在可接受的质量下，本机 TTS 无法超过实时速度 —— int4 本是能补上差距的杠杆，但没有通过听测。约 1.4 倍 + 流式输出即为如实的稳态。

## 环境要求
- NVIDIA **GB10 系统**（如 DGX Spark）、驱动 + **NVIDIA Container Toolkit**（`--gpus all` 可用）、**Docker**
- **约 100 GB 可用磁盘空间**用于权重
- 建议**无头运行**（关闭屏幕、仅远程/SSH）以获得最大内存余量

## 1. 下载权重（Hugging Face）
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download InquiringMinds-AI/LongCat-Next-w8a8-int8-GB10 --local-dir ./longcat-next-gb10-weights
```
权重目录是**自包含的**（约 90 GB）：量化主干 + 分词器 + 图像解码器 + 音频声码器。无需额外下载。

随后提取生成头所需的码本嵌入附件（一次性，约 931 MB，从分片中已有的未量化 `embed_tokens` 切出）：
```bash
python3 quantize/extract_codebook_embeddings.py ./longcat-next-gb10-weights
```
缺少 `codebook_embeddings.safetensors` 时服务仍可启动，但音频/图像生成头的前级码本条件会静默退化为零向量
（留意日志中的 `codebook_embeddings.safetensors not found`）。

## 2. 构建镜像
```bash
docker build -t longcat-next-gb10 .
```
在 `lmsysorg/sglang:v0.5.16-cu130` 基础镜像上叠加 LongCat-Next 适配层与 GB10 修复（基础镜像的拉取是此处唯一较大的下载）。

## 3. 启动服务
```bash
./run.sh ./longcat-next-gb10-weights
```
首次启动需加载约 90 GB（数分钟）。当看到 `The server is fired up and ready to roll!` 时，
API 已在 `http://localhost:8090` 上提供服务，并**在所有模态上兼容 OpenAI 接口**（可直接用 `openai` SDK / LangChain）。`run.sh` 会转发全部 `LCN_*`/调优环境变量，因此 `LCN_YARN=1 ./run.sh …` 按预期生效。

同时也保留 SGLang 原生 `/generate` 接口（透传）；随附的 `gen_*`/`understand_*` 脚本即使用它。
生成的文件**仅在 `LCN_KEEP_ARTIFACTS=1` 时**保留在 `./outputs/` —— 默认发送给客户端后即删除。

## 选择服务配置方案

| 你要服务的是… | 设置 | 得到什么 |
|---|---|---|
| 全部能力（默认） | *（无需设置）* | 所有模态；完整原生 128k 上下文；生成头在首次使用时惰性分配约 22 GB（或用 `LCN_PREWARM=1` 在启动时预热） |
| agent / 纯文本客户端（Claude Code、工具循环） | `LCN_AGENT=1` | 生成接口返回 403，约 22 GB 永不分配；理解能力保留；前缀缓存令重复系统提示词近乎免费 |
| 多个长会话并发 | `LCN_AGENT=1 MAX_TOTAL_TOKENS=917504 MEM_FRACTION=0.88` | **约 6 个完整 128k 上下文**的 KV 池（实测：800 557 token，稳态余量 7.5 GB，36k-token 提示词以约 1.8k tok/s 预填充且检索准确） |
| 256k 上下文 | `LCN_YARN=1` | RoPE-YaRN ×2；因 YaRN 可能轻微影响短上下文/生成质量而设为可选 |

KV 经 MLA 压缩，仅 **约 31.5 KB/token** —— 一个完整 128k 上下文只占 3.94 GB，多上下文方案因此可行。默认 `MEM_FRACTION=0.72` 已能容纳完整的 131 072-token 池；本机真正的约束是生成头所需的*物理*内存余量，而非该比例。**不要随意调高 `MEM_FRACTION`** —— GB10 在内存耗尽时会直接断电关机（需要按物理电源键），没有优雅的 OOM。

## 安全

服务本身**不带内置认证**，因此默认配置将其隔离于网络之外：

- **默认仅回环。** `run.sh` 与 `docker-compose.yml` 只把端口发布到 `127.0.0.1:8090` —— 宿主机可达，局域网不可达。
- **如需对网络开放**，同时设置监听地址与密钥：
  ```bash
  LCN_BIND=0.0.0.0 LCN_API_KEY=$(openssl rand -hex 24) ./run.sh ./longcat-next-gb10-weights
  ```
  设置 `LCN_API_KEY` 后，除 `GET /health` 外的所有接口都要求 `Authorization: Bearer <key>`。
  （在非回环地址上未设密钥时，`run.sh` 会给出警告。）
- **SGLang 原生管理面不对外。** 透传代理为默认拒绝：仅推理/只读接口（`/generate`、`/get_model_info`、`/v1/models` 等）放行；变更类控制接口（`/flush_cache`、`/update_weights*`、性能分析等）返回 `404`。
- **TTS 参考音频路径受限。** 自定义 `voice` 路径必须位于内置 voices 目录或挂载的输出目录（或 `LCN_VOICE_DIR`）之下；任意容器路径会被拒绝。

## 4. 各模态测试（OpenAI 接口）

**文本**
```bash
curl -s localhost:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"longcat-next","messages":[{"role":"user","content":"Name two oceans."}],"max_tokens":24}'
```

**图像生成**（返回 base64 PNG，OpenAI images 格式）
```bash
curl -s localhost:8090/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"A photograph of a red apple on a wooden table.","response_format":"b64_json"}'
```

**声音克隆 TTS**（返回 audio/wav；`voice`=`en`|`zh`|容器内路径）
```bash
curl -s localhost:8090/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"The quick brown fox jumps over the lazy dog.","voice":"en"}' -o speech.wav
```
加上 **`"stream": true`** 可以边生成边接收音频（带流式头的 WAV，或配合 `"response_format":"pcm"` 输出原始 s16le/24 kHz PCM）—— 首段音频约 6.5 秒即达，而非等整段完成。**多句输入会完整朗读**：模型逐句分轮朗读，文本读完后自行停止（服务侧防护重复朗读、擅自续写与失控轮次 —— 朗读语义仅施加于 TTS 形态的请求，开放式语音生成不受约束）。

**图像 / 视频 / 音频理解** —— 向 `/v1/chat/completions` 发送带 `image_url`、`video_url` 或 `input_audio` 内容块的标准 OpenAI 多模态消息，例如：
```bash
curl -s localhost:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"longcat-next","max_tokens":80,
  "messages":[{"role":"user","content":[
    {"type":"text","text":"Describe this image."},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,<BASE64>"}}]}]}'
```

> 另有随附脚本 `gen_image.py`、`gen_audio.py`、`understand_video.py`（位于 `/workspace/scripts/`）及 SGLang 原生 `/generate` 接口。文本对话支持 **`stream: true`**（SSE），与 OpenAI API 一致。

## 自检

在你的机器上端到端验证每个模态：
```bash
docker exec longcat-next python3 /workspace/scripts/selftest.py
```
输出文本、图像生成、图像理解、音频生成、音频理解、视频理解与工具调用的 PASS/FAIL；任一失败则以非零码退出。生成类检查的 PASS 只代表输出*格式正确* —— 在信任一次改动之前，请亲自看图、亲耳听音频。

## Agent 模式与 Claude Code

`LCN_AGENT=1` 用生成头换取确定的内存余量：图像/音频*生成*接口返回 403（其约 22 GB 永不分配）。图像/音频/视频**输入的理解能力保留**。配合前缀缓存，这是每轮都重发大段系统提示词的 agent 客户端的推荐配置 —— 实测：15.6k-token 前缀冷态约 5.9 秒、**热态约 0.36 秒**（16 倍）。多会话并发时叠加[选择服务配置方案](#选择服务配置方案)中的多上下文方案。

网关同时支持 **Anthropic Messages API**（`POST /v1/messages`，含流式与工具调用），Anthropic 原生客户端可直接使用。Claude Code：

```bash
export ANTHROPIC_BASE_URL=http://<host>:8090
export ANTHROPIC_AUTH_TOKEN=<your LCN_API_KEY>
export ANTHROPIC_MODEL=longcat-next
export ANTHROPIC_SMALL_FAST_MODEL=longcat-next
claude
```

如实评估：模型能完成真实的 agent 工具循环（读写/执行、错误恢复、多轮），但毕竟是 75B-A3B —— 长随机字符串上偶有路径/参数笔误，请复核其工作。`test/test_anthropic.py` 可端到端自测该路由。

## 服务运维（人与智能体通用）

**`GET /health`** —— 后端就绪前返回 `503 {"status":"loading"}`，就绪后返回 `200 {"status":"ok"}`。永不要求认证。

**`GET /status`** —— *实际生效*配置的机器可读权威来源：运行中进程眼里的每个调优开关（而非启动脚本的意图），外加预热状态。凡是"这台机器到底跑的什么配置"之类的疑问，读 `/status`，别读启动脚本。

**`LCN_PREWARM=1`** 在启动时各执行一次真实的图像生成与短 TTS，把约 22 GB 的惰性头分配和首调用编译成本提前付清。`/status → prewarm.status` 依次为 `pending → running → ready`；**`degraded` 表示某条生成路径预热失败** —— 服务仍可用，但应视为该路径已损坏，而非普通警告。

**故障特征 → 原因：**

| 现象 | 原因 / 处置 |
|---|---|
| 任意接口返回 `503 backend unavailable` | 仍在加载（冷启动约 5–8 分钟）；轮询 `/health` |
| 图像/语音接口返回 `403` | `LCN_AGENT=1` —— 生成被配置禁用，不是故障 |
| 首张图约 3.5 分钟、之后约 2.4 分钟 | 首调用承担头分配 + 图捕获；用 `LCN_PREWARM=1` 前置该成本 |
| 整机断电关机（需按物理电源键） | 内存耗尽 —— 降低野心，绝不要靠调高 `MEM_FRACTION` "解决"容量；无头运行；不要并行其他 GPU 负载 |
| `./outputs/` 里找不到生成文件 | 设计如此 —— 除非 `LCN_KEEP_ARTIFACTS=1`，产物发送后即删除 |

## 调优（环境变量）

全部由 `run.sh` 转发，并可从 `GET /status` 读回实际生效值。括号内为默认值。

**容量 / 配置方案**
- `MEM_FRACTION`（0.72；YaRN 下 0.74；agent 配置 +0.03）—— SGLang 静态显存比例。调高前先看上文断电警告。
- `MAX_TOTAL_TOKENS`（131072；YaRN 下 262144）—— KV 池大小；`917504` + `MEM_FRACTION=0.88` + `LCN_AGENT=1` 为实测多上下文方案。
- `LCN_AGENT`（0）—— agent 配置：生成接口 403，头内存永不分配。
- `LCN_YARN`（0）—— YaRN ×2 扩展至 256k 上下文。
- `LCN_RADIX`（1）—— 前缀（radix）缓存。依赖入口脚本默认的 `expandable_segments` 分配器才能在统一内存上不泄漏。
- `LCN_KV_DTYPE`（未设）—— 如 `fp8_e4m3` 将 KV 字节减半 → 同比例下约 2 倍 token 容量；先在你的负载上验证质量。

**速度（均为实测；测量过程见 FINDINGS）**
- `LCN_CUDAGRAPH`（1）+ `LCN_CUDAGRAPH_BS`（32）—— 解码 CUDA 图。单流 +6 %，16 并发聚合 +13.6 %。设 `0` 恢复 eager。
- `LCN_HEAD_GRAPH`（1）—— 生成头逐层 CUDA 图重放。数学上完全一致（捕获时重放与 eager 逐位比对；任何捕获失败自动永久回退 eager）。
- `LCN_INT8_HEADS`（`audio`）—— 逐头 int8 FFN：`audio` | `visual` | `both` | `0`。`audio` 为经判定的默认（TTS 每帧 −34–36 %，听测无差异）；`both` 换取约 8 % 图像提速，代价是实测的空间质量下降；`audio4`（int4）存在但听测被否决。
- `LCN_NGRAM`（0）—— N-gram 投机解码（链式草稿，`LCN_NGRAM_DRAFT`=4）。可选；与生成共存（生成批次期间投机暂停，结束后恢复）。

**图像生成**
- `LCN_REFINER_CFG_RANGE`（`1.0,0.0` = 关闭引导项，每图 −17 %，输出经判定无差异；`0.0,1.0` 恢复原引导精修）· `REFINER_STEPS`（10；最高保真可升至 28，延迟约 1.5 倍）· `IMAGE_GEN_CFG_SCALE`（3.0）· `IMAGE_GEN_TEMPERATURE` / `IMAGE_GEN_TOP_K` / `IMAGE_GEN_TOP_P`。

**TTS**
- `LCN_TTS_STREAM`（1）—— 滑窗分块声码 + PCM 流式输出；最终 `.wav` 由流式字节原样拼装。
- `LCN_TTS_MULTI`（1）—— 多句分轮朗读（见 TTS 一节）。设 `0` 恢复单轮。
- `LCN_TTS_SILENCE_FRAMES`（1）与 `LCN_TTS_TRIM_LEAD_MS`（150）—— 起音调理：吸收模型首帧噪音，再裁掉渲染出的前导静音。
- `LCN_TTS_TRIM_TAIL_MS`（250）—— 模型会把尾部静音*当作内容*生成；拼装后的 wav 尾部裁至该时长。设 `0` 关闭。
- `AUDIO_GEN_TEMPERATURE` / `AUDIO_GEN_TOP_K`。

**运维**
- `LCN_PREWARM`（0）—— 启动时预热两条生成路径；见"服务运维"。
- `LCN_KEEP_ARTIFACTS`（0）—— 保留生成文件而非发送后删除。
- `LCN_VERBOSE`（0）—— 逐步生成调试日志。
- `LCN_MODEL_NAME`（`longcat-next`）、`LCN_VOICE_DIR`、`LCN_NGRAM_EOS`（−1，旧式哈希 —— 该权重即在此语义下训练）。

## 统一内存上的内存稳定性（重要）

容器/启动脚本已内置两项修复 —— 自写启动器时务必保留：

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**（入口脚本默认）。缺少它时，视觉编码器会在长的多图对话中让 CUDA 缓存分配器碎片化，且碎片段永不归还 —— 在统一内存机器（GB10）上会持续吞噬*系统*内存，直至整机僵死或 OOM killer 出手。80 轮多图浸泡实测：无该标志 −7.6 GB 且持续下降，有则 −2.9 GB 后收敛持平。
- **`--shm-size=32g`**（run.sh / compose / 你自己的 `docker run`）。SGLang 通过 `/dev/shm` 在进程间传递多模态像素张量；Docker 默认的 64 MB 会在首个多图请求上以 SIGBUS 崩溃。tmpfs 按需分配 —— 上限设高不花钱。

## 示例输出

见 `examples/`：一张示例生成图像与一段声音克隆片段，供下载权重前了解预期质量。

## 备注
- **针对无头 GB10 运行优化** —— 关屏/仅远程运行以获得最大内存余量。
- **音频时长由模型决定** —— 输出随文本需要而定；约 40 秒（1000 帧）的保险上限仅用于防失控。
- **图像质量校准**：简单构图可靠；密集场景（满座咖啡馆、人群）会暴露该量化下模型的构图上限。这是权重本身的能力边界，不是服务栈的问题 —— 此处每个服务默认项都以"不使其更差"为门槛验证过。

## 仓库结构

```
.                       可直接运行的包（本 README、Dockerfile、run.sh 等）
├── gateway.py          兼容 OpenAI 的网关，前置于 SGLang（全模态 + 工具 + 流式 TTS）
├── anthropic_route.py  Anthropic Messages API（/v1/messages）—— Claude Code 等
├── longcat_tools.py    工具调用：TS 命名空间提示构建 + <longcat_tool_call> 解析（两种语法）
├── entrypoint.sh       SGLang + 网关进程守护 + 调优默认值所在地
├── new_files/          LongCat-Next 的 SGLang 适配层（models / layers / processors，
│                       含 int8_head_ffn.py 与 lcn_head_graph.py —— 实测提速杠杆）
├── patches/            容器构建补丁
├── quantize/           w8a8_int8 导出工具（权重的制作方式）
├── test/               selftest.py + 各模态示例客户端 + 单元门禁
├── voices/             TTS 参考音频（en：公有领域 LibriVox；zh：美团 MIT）
├── examples/           示例生成图像 + 语音片段
└── research/           构建历程 —— 工程叙事 + 验证工具
    ├── FINDINGS.md       完整历程：bug、修复、基准、被推翻的结论（从这里读起）
    ├── int8_heads/       头量化与启动延迟的基准与追踪分析器
    ├── tts_streaming/    分块声码质量门禁工具
    └── oracle/           bnb-int8 能力证明 + 可靠性探针
```

## 致谢与许可
模型：**美团 LongCat-Next**（MIT）。推理栈：**SGLang**。英文示例声音：公有领域 **LibriVox** 朗读。中文示例声音：美团 LongCat 示例片段（MIT）。见 [LICENSE](LICENSE)。
