<!-- LANG -->
[English](README.md) | **中文**

# 在 DGX Spark (GB10) 上运行 LongCat-Next —— 全模态推理服务

在单台 **NVIDIA GB10 系统（`sm_121`）** 上，通过**单个 SGLang 进程**运行[美团 **LongCat-Next**](https://huggingface.co/meituan-longcat)（75B 总参 / ~A3B 激活的任意到任意多模态 MoE），并提供**兼容 OpenAI 的全模态接口**：

| 能力 | OpenAI 接口 | 状态 |
|---|---|:--:|
| 文本生成（支持 `stream`） | `POST /v1/chat/completions` | ✅ |
| 图像 / 音频 / 视频**理解** | `POST /v1/chat/completions` | ✅ |
| **图像生成**（文本 → 图像） | `POST /v1/images/generations` | ✅ |
| **声音克隆音频生成** | `POST /v1/audio/speech` | ✅ |
| 工具 / 函数调用 | `POST /v1/chat/completions`（`tools`） | ✅ |

<sub>（LongCat-Next 不支持视频**生成**，视频仅支持理解。）</sub>

量化为 **`w8a8_int8`**（8-bit 权重 + 逐 token int8 激活）—— 切换到 8-bit 正是图像与音频生成变得连贯的关键（4-bit 下两者都崩溃）。我们未在后续修复之后重新测试 4-bit，因此 8-bit 是*经过验证*的设置，而非已证实的下限。单个自包含约 90 GB 模型，在单台 GB10 上稳定运行，并通过 [7/7 自检](#自检)端到端验证。

> **构建历程** —— 让全部模态跑通，需要解开两个看似同一症状（生成不连贯）的独立问题：切换到 8-bit，以及服务流程中的一处**结构性**修复（被丢弃的空间锚点）—— 外加一次发现了静默 MoE 缩放 bug 的对抗式多智能体评审。完整历程见 **[research/FINDINGS.md](research/FINDINGS.md)**。

> 专为 GB10 超级芯片（`sm_121`）构建 —— 已在 DGX Spark 上验证，预期可在任何基于 GB10 的机器上运行
> （依赖的是芯片而非具体产品）。只有 cu130 的 SGLang 基础镜像能为 `sm_121` 编译 Triton；在其他 GPU 上不保证运行。

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

## 2. 构建镜像
```bash
docker build -t longcat-next-gb10 .
```
在 `lmsysorg/sglang:v0.5.12.post1-cu130` 基础镜像上叠加 LongCat-Next 适配层与 GB10 修复（基础镜像的拉取是此处唯一较大的下载）。

## 3. 启动服务
```bash
./run.sh ./longcat-next-gb10-weights
```
首次启动需加载约 90 GB（数分钟）。当看到 `The server is fired up and ready to roll!` 时，
API 已在 `http://localhost:8090` 上提供服务，并**在所有模态上兼容 OpenAI 接口**（可直接用 `openai` SDK / LangChain）：

| 模态 | OpenAI 接口 |
|---|---|
| 文本 | `POST /v1/chat/completions` |
| 图像 / 视频 / 音频**理解** | `POST /v1/chat/completions`（`image_url` / `video_url` / `input_audio` 内容块）|
| **图像生成** | `POST /v1/images/generations` |
| **声音克隆 TTS** | `POST /v1/audio/speech`（`voice`：`en`、`zh`，或容器内参考音频路径）|

同时也保留 SGLang 原生 `/generate` 接口（透传）；随附的 `gen_*`/`understand_*` 脚本即使用它。
生成的文件保存在 `./outputs/`。

## 安全

本服务**不内置鉴权**，因此默认配置将其限制在本机、不暴露到网络：

- **默认仅回环。** `run.sh` 与 `docker-compose.yml` 默认仅在 `127.0.0.1:8090` 上发布端口 ——
  仅本机可访问，局域网不可达。
- **如需暴露到网络**，请同时设置监听接口与密钥：
  ```bash
  LCN_BIND=0.0.0.0 LCN_API_KEY=$(openssl rand -hex 24) ./run.sh ./longcat-next-gb10-weights
  ```
  设置 `LCN_API_KEY` 后，除 `GET /health` 外的所有接口都要求 `Authorization: Bearer <key>`。
  （若在非回环接口上发布却未设密钥，`run.sh` 会发出警告。）
- **不暴露 SGLang 原生管理接口。** 透传代理为默认拒绝：仅推理/只读接口
  （`/generate`、`/get_model_info`、`/v1/models` 等）放行；改写型控制接口
  （`/flush_cache`、`/update_weights*`、性能分析等）一律返回 `404`。
- **TTS 参考音频路径受限。** 自定义 `voice` 路径必须位于内置 voices 目录或挂载的输出目录之下
  （或 `LCN_VOICE_DIR`）；任意容器路径将被拒绝。

## 4. 测试各模态（OpenAI 接口）

**文本**
```bash
curl -s localhost:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"longcat-next","messages":[{"role":"user","content":"列举两个大洋。"}],"max_tokens":24}'
```

**图像生成**（返回 base64 PNG，OpenAI images 格式）
```bash
curl -s localhost:8090/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"一只橘色的小猫坐在窗台上。","response_format":"b64_json"}'
```

**声音克隆 TTS**（返回 audio/wav；`voice`=`en`|`zh`|容器内路径）
```bash
curl -s localhost:8090/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"今天天气很好，我们一起去公园散步吧。","voice":"zh"}' -o speech.wav
```

**图像 / 视频 / 音频理解** —— 用带 `image_url`、`video_url` 或 `input_audio` 内容块的
`/v1/chat/completions`（标准 OpenAI 多模态消息），例如：
```bash
curl -s localhost:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"longcat-next","max_tokens":80,
  "messages":[{"role":"user","content":[
    {"type":"text","text":"描述这张图片。"},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,<BASE64>"}}]}]}'
```

> 也可使用随附脚本 `gen_image.py`、`gen_audio.py`、`understand_video.py`（位于 `/workspace/scripts/`）
> 以及 SGLang 原生 `/generate` 接口。文本对话支持 **`stream: true`**（SSE），与 OpenAI 接口一致。

## 自检

在你的机器上端到端验证所有模态：
```bash
docker exec longcat-next python3 /workspace/scripts/selftest.py
```
逐项打印文本、图像生成、图像理解、音频生成、音频理解、视频理解的 PASS/FAIL；任一失败则以非零码退出。

## 上下文长度

默认即模型**原生 128k**（`max_total_tokens` 131072）。MLA 使 KV 缓存非常廉价（约 16 KB/token），
因此长上下文几乎免费 —— 限制因素只是 `--mem-fraction-static`，已设为可容纳完整 128k 池。

设置 **`LCN_YARN=1`** 可通过 YaRN（RoPE 因子 2）扩展到 **256k**。这是可选项，因为 YaRN 可能略微
影响短上下文 / 生成质量；默认保留未缩放的 128k 路径。两种模式都通过自检，且都稳妥低于 GB10 的显存余量
（完整生成过程中 128k 峰值约 95 GB，256k 约 101 GB）。已验证：一段 28k token 的提示能正确召回置于开头的事实。

## 调参（环境变量）

在 `docker run -e …`（或 `docker-compose.yml`）中设置：
`MEM_FRACTION`（0.72；`LCN_YARN` 下为 0.74）、`MAX_TOTAL_TOKENS`（131072；`LCN_YARN` 下为 262144）、
`LCN_YARN`（0）、`IMAGE_GEN_CFG_SCALE`（3.0）、
`IMAGE_GEN_TEMPERATURE`/`IMAGE_GEN_TOP_K`/`IMAGE_GEN_TOP_P`、`AUDIO_GEN_TEMPERATURE`/`AUDIO_GEN_TOP_K`、
`REFINER_STEPS`（10；调高至 28 可获得最高图像保真度，延迟约 1.5 倍），
以及 `LCN_VERBOSE=1`（逐步调试日志）。

## 示例输出

`examples/` 目录中提供了一张示例生成图像与一段声音克隆音频，方便你在下载权重前了解预期质量。

## 常见问题

- **冷启动约 5–8 分钟**（加载约 90 GB）。就绪前 `GET /health` 返回 `503 {"status":"loading"}`，
  就绪后返回 `200 {"status":"ok"}`；任何接口返回 `503 "backend unavailable"` 表示仍在加载。
- **运行中整机断电** → GPU 显存占用过高；请无头运行、不要同时跑其他重度 GPU 任务、不要调高 `MEM_FRACTION`。
- **首张图像较慢（约 4–5 分钟）** —— 1369 个视觉 token + 扩散精修；音频接近实时。

## 注意事项
- **面向无头 GB10 运行优化**（发布时）—— 请以关屏 / 纯远程方式运行，以获得最大显存余量。
- **音频时长由模型决定** —— 输出长度匹配文本所需，没有任务长度下限；仅有约 40 秒（1000 帧）的安全上限用于防止失控生成。
- **首张图像**约需 4–5 分钟（1369 个视觉 token + 扩散精修）；音频接近实时。

## 致谢与许可
模型：**美团 LongCat-Next**（MIT）。推理框架：**SGLang**。英文示例音色：公有领域 **LibriVox** 朗读。
中文示例音色：美团 LongCat 示例片段 spk_syn.wav（MIT）。详见 [LICENSE](LICENSE)。
