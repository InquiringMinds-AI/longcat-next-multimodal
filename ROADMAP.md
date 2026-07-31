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

- [x] Survey upstream SGLang main + releases for LongCat-Next / longcat_flash changes
      (2026-07-31: upstream v0.5.16 has the longcat_flash TEXT lineage + its own leaner
      n_gram_embedding — still ZERO multimodal Next support; the overlay stays necessary.
      v0.5.16-cu130 image published 2026-07-25; drift bounded: 166 diff-lines in
      longcat_flash.py across the four releases vs our ~178-line overlay delta.)
- [x] **Decision: REBASE to v0.5.16-cu130** — front-loads the base so #3 and #4 land on
      the final foundation; inherits two months of engine fixes; port risk bounded by the
      test battery (selftest, soak, Anthropic tests, live Claude Code loop).
- [ ] Port overlay to v0.5.16, full revalidation — **IN PROGRESS, BLOCKED on a TTS
      quality regression** (2026-07-31). Port state: three-way merges done; eos_token_id
      wired through (env `LCN_NGRAM_EOS`, -1 = legacy hashing); Claude-imitation tool
      dialect parser added after live CC runs surfaced it; scipy pinned 1.17.1.
      Validation: automated battery ALL GREEN (7/7 selftest, 6/6 degeneracy, 5/5
      anthropic, 3/3 live CC, 40-turn soak converging-flat) — but HUMAN review caught
      generation regressions the battery could not: (a) image quality — CLOSED
      (2026-07-31, paired comparison + owner review): the rebase regression itself was
      FIXED — root cause was the v0.5.16 n-gram EOS-break (neutralized via
      LCN_NGRAM_EOS=-1; keep -1 the default: the checkpoint was trained under
      cross-boundary hashing). The residual hard-composition failures (object-
      interaction geometry, body-vehicle fusion, floating subjects) reproduce on
      v0.5.12 with the same prompts — NOT a rebase bug. Owner verdict on the paired
      set: no image is fully flawless (even the best, a portrait, has a subtle
      geometry flaw); most likely an unavoidable quant artifact, though a serving-
      implementation fault common to BOTH builds remains plausible (the paired test
      only rules out rebase-introduced regression), and whether the BF16 base weights
      could do better is an actual unknown (BF16 has never been run — no hardware on
      hand can). Document as a model/quant capability bound; (b) TTS
      voice-clone — STILL REGRESSED: garbled first word ("Self"->"helf/pulf/uf-") +
      subtly robotic prosody throughout. ELIMINATED: n-gram eos (both paths verified
      env-covered), scipy 1.18, our overlay code (identical), mm offset machinery
      (agent-diffed byte-identical across tags). 2026-07-31 paired-probe evidence
      (temp-0 + teacher-forced captures, radix off, both engines):
      (i) SAMPLER statically near-cleared — sglang's joint top-k/top-p Python path is
      behavior-equivalent across versions; flashinfer 0.6.11->0.6.14's new top_k_first
      fast path doesn't engage for filter_apply_order="joint"; kernel diffs are two
      edge-case fixes. temp-0 wav pairs (no sampling kernels at all) delivered for
      owner ears — if those still garble, sampler is conclusively out.
      (ii) BACKBONE PREFILL BIT-IDENTICAL across versions (teacher-forced scoring of a
      fixed 127-token passage: zero logprob delta), and each version is bit-exact
      run-to-run on text. BUT DECODE NUMERICS DIFFER: greedy prose diverges at token
      44/300; pre-divergence |dlogprob| max 0.268 mean 0.028. A decode-path kernel
      changed (flashinfer decode attn / fused-MoE / DeepGemm) — plausible carrier for
      whole-clip prosody drift (audio codec tokens decode-generated + perturbation-
      sensitive). Toggle A/B running: SGLANG_ENABLE_JIT_DEEPGEMM=0 on v0516 vs the
      v512 baseline.
      (iii) TTS path is NONDETERMINISTIC even at temp 0 (main-stream greedy; run
      lengths differ 38/53 v512, 82/109 v516) while text is bit-exact — the audio
      codec heads sample internally; nondeterminism enters via audio machinery only.
      (iv) NEW OBJECTIVE DELTA: same sentence/ref, v512 clips run 2.2-3.4s and their
      transcripts STOP AT THE FIRST CLAUSE; v516 clips run 5.1-7.3s and speak further.
      Audio-end behavior differs across versions (possibly the latent half-open-vs-
      inclusive mm offset issue) — v516 may actually be MORE complete here.
      Remaining suspects, reordered: decode-numerics drift (toggle test running),
      NgramEmbeddingManager token-table lifecycle (new ne_skip_token_table_update
      path at the chunked-prefill->decode boundary — static content-equivalent but
      overlap-timing-sensitive per upstream's own comment). ALSO FOUND (latent, both versions): overlay emits
      half-open mm offsets where upstream treats them inclusive -> pad clobbers
      <longcat_audio_end>; fix offsets to inclusive when touching this area.
      v0.5.12 image remains :latest / shipping; v0.5.16 work lives in :v0516.

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

## 5. Performance experiments (after #3/#4 and ALL fixes — no moving targets)

- [ ] **mem-fraction headroom mapping (all-modality)**: 0.72 is inherited, not derived —
      each +0.01 buys ~40k KV tokens here. With gen heads warmed, drive image-gen +
      multi-image understanding while sampling MemAvailable at 1s; step 0.72→0.73→0.74
      until the measured floor drops below ~2.5–3GB margin; ship the last safe value.

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
