# Roadmap

Improvement agenda following the 2026-07-30 stability + agent campaign (see
`research/FINDINGS.md`). Ordered so earlier work is never invalidated by later work:
foundation decisions first, engine-deep changes before performance tuning (they shift the
numbers), measurements before documentation, docs last.

## 1. Base-version decision (foundation — everything patches against this)

Check whether SGLang upstream has grown native or partial LongCat-Next support since
v0.5.12.post1, and whether newer releases change the overlay surface (model registry,
multimodal processors, MoE runner, triton version). **Decide: stay on 0.5.12 or rebase.**
Everything below patches files whose location and content depend on this choice — a later
rebase would force re-porting, so this gate comes first.

- [ ] Survey upstream SGLang main + releases for LongCat-Next / longcat_flash changes
- [ ] Decision recorded here; if rebasing: port overlay, re-run selftest + soaks

## 2. Incremental streaming (orthogonal — gateway only, no relaunch risk)

Both the OpenAI tool path and the Anthropic route currently buffer the whole completion
(~20 s of dead air for a 400-token answer at ~21 tok/s). Stream tokens through live;
start buffering only when `<longcat_tool_call>` appears mid-stream; emit parsed tool
calls at the end. Applies to `/v1/chat/completions` (tools branch) and `/v1/messages`.
No interaction with any engine work below.

- [ ] OpenAI route: stream-with-tool-detection
- [ ] Anthropic route: real SSE deltas (replace buffered-then-emitted synthesis)
- [ ] test_anthropic streaming check tightened (multiple text deltas expected)

## 3. n-gram embedding rework (engine-deep — the double unlock)

Both abandoned speed levers failed in the same file: CUDA graph capture and NGRAM
speculative decoding each die in `n_gram_embedding.py` forward, which contains
`if ignored_mask.any():` — a host-side sync + dynamic branch in the decode path
(capture poison) and history-hash indexing that draft positions violate (illegal
memory access). Make the layer branch-free and draft-position-safe.

- [ ] Tensorize the ignored-mask path (no `.any()`, no data-dependent branching)
- [ ] Audit gather/hash indexing for out-of-history positions (clamp or mask)
- [ ] Re-test CUDA graph capture (`LCN_CUDAGRAPH=1`) — currently fails even at bs 8
- [ ] Re-test NGRAM spec decode (`LCN_NGRAM=1`) — currently illegal-access faults
- [ ] Quality check: outputs identical pre/post rework (temp-0 diff on fixed prompts)

## 4. MoE kernel tuning (after #3 — tune the final decode path once)

Every launch warns: default fused-MoE Triton config for
`E=256,N=1024,device_name=NVIDIA_GB10,dtype=int8_w8a8,per_channel_quant=True` (and the
`_down` variant) is missing. Decode (~21 tok/s bf16) is MoE-GEMM-bound. Run a targeted
sweep for exactly these shapes and bake the JSONs into the image. Done after #3 so the
tuned path is the shipping path (config files are also keyed by triton version — another
reason #1 settles first).

- [ ] Targeted `tuning_fused_moe_triton_sep.py` sweep (both up and down proj; artifacts
      on a mounted volume, never in an `--rm` container)
- [ ] Ship configs under `patches/`, COPY into the image's triton config dir
- [ ] Before/after decode bench (same 3-workload suite)

## 5. Performance experiments (after #3/#4 — otherwise measured twice)

- [ ] **DeepGemm accuracy flag**: launches warn `scale_fmt is not ue8m0 — might cause
      accuracy degradation on Blackwell`. A/B DeepGemm on/off for quality + speed;
      keep or disable with evidence.
- [ ] **Prefill tuning**: cold prefill measured ~2.6k tok/s; try larger
      `chunked_prefill_size` / `max_prefill_tokens` on the 128 GB box.
- [ ] **fp8 KV re-bench** post-#4 (owner-validated for quality; −41% decode pre-tuning —
      the gap may narrow once MoE kernels stop dominating)

## 6. Operational polish

- [ ] **`LCN_PREWARM=1`**: opt-in startup warmup of the image/audio generation heads —
      moves the ~25 GB lazy allocation and the 4–5 min first-image surprise to load
      time, where the operator expects cost.
- [ ] **Processor registration cleanup**: `AutoProcessor build failed → AutoTokenizer
      fallback` and deprecated `image_processor_class` mappings work today but are
      fragility against future transformers bumps in the base image.

## 7. Final numbers + legibility (last — documents whatever the above produced)

- [ ] Re-run the full bench suite on the final configuration; update README numbers
- [ ] README: lead with the Claude Code / agent-mode headline (currently mid-file)
- [ ] HF model card: agent mode + Anthropic route + updated serving guidance
- [ ] Short terminal recording of Claude Code driving the container
