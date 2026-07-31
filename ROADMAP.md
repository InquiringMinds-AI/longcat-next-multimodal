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
      sensitive). Toggle A/B DONE: SGLANG_ENABLE_JIT_DEEPGEMM=0 on v0516 is
      BIT-IDENTICAL to DeepGemm-on (300/300 greedy tokens, zero scoring drift) —
      DeepGemm is INERT for this w8a8_int8 checkpoint; the startup ue8m0 warning is
      noise. DeepGemm ELIMINATED as a TTS suspect; decode drift must come from
      flashinfer decode-attention or fused-MoE changes between releases.
      (iii) TTS path is NONDETERMINISTIC even at temp 0 (main-stream greedy; run
      lengths differ 38/53 v512, 82/109 v516) while text is bit-exact — the audio
      codec heads sample internally; nondeterminism enters via audio machinery only.
      (iv) NEW OBJECTIVE DELTA: same sentence/ref, v512 clips run 2.2-3.4s and their
      transcripts STOP AT THE FIRST CLAUSE; v516 clips run 5.1-7.3s and speak further.
      Audio-end behavior differs across versions (possibly the latent half-open-vs-
      inclusive mm offset issue) — v516 may actually be MORE complete here.
      OWNER EAR VERDICT on the six paired clips (2026-07-31) — REFRAMES THE BLOCKER:
      every v512 clip SKIPS the word "Self" entirely (silent omission of the first
      word); v516 clips damage but attempt the onset ("elf reflection" / "coal. Self
      reflection" — junk syllable then the full correct word / "helf reflection");
      NO prosodic issues in ANY of the six. Conclusions: (1) the onset defect is a
      SHARED audio-start boundary bug present in BOTH builds in different forms —
      v512's silent word-drop is arguably worse than v516's audible garble; combined
      with the end-behavior delta (v512 clause-truncates, v516 speaks further), v516
      is MORE faithful on both clip edges. (2) The "robotic prosody throughout" from
      the original review DID NOT REPRODUCE — and the owner subsequently clarified
      (2026-07-31) that "robotic" meant TEMPO/CADENCE, not tone: careful radio-
      broadcast-style word separation, "probably a prosodic effect of the words being
      used rather than a failure." The prosody arm of the regression is thereby
      WITHDRAWN as a serving defect. (Owner corollary: test texts like "all systems
      nominal" carry strong non-conversational register priors — broadcast cadence is
      faithful rendering, not a fault. Methodology: use conversational-register
      sentences for ear checks unless register itself is under test.) Radix-ON v516
      captures (temp-0 + cold/warm prod pair) still in flight as confirmation that
      the default config is also clean.
      If prosody stays clean: rebase quality-UNBLOCKED; the shared onset bug becomes
      its own roadmap item (prime suspect: the latent half-open-vs-inclusive mm
      offset handling at the audio boundaries, plus cold n-gram history at
      generation start). ALSO FOUND (latent, both versions): overlay emits
      half-open mm offsets where upstream treats them inclusive -> pad clobbers
      <longcat_audio_end>; fix offsets to inclusive when touching this area.
      v0.5.12 image remains :latest / shipping; v0.5.16 work lives in :v0516.

## 1b. MISSING CODEBOOK EMBEDDING SIDECAR (found 2026-07-31 — affects BOTH builds,
##     both generation modalities, since original stand-up)

Investigating the shared TTS onset defect surfaced that
`codebook_embeddings.safetensors` never existed in the serving weights dir. The
overlay's `_codebook_embed_fn` / `_embed_multimodal_ids` fall back to ZERO vectors
for codebook-range ids when the sidecar is absent (the "will be clamped" warning
undersells it — the log shows the warning firing ~8k times per generation). Effect:
in `CasualDepthTransformerHead`, the within-frame prior-level conditioning
(cumsum of prior codebook embeddings) is all zeros — every codebook level beyond
level 0 samples with no knowledge of what the previous levels chose, for BOTH the
audio head (8 levels) and visual head (8 levels). A standing quality ceiling on both
generation paths that no automated battery could see, and a candidate contributor to
the shared TTS onset damage and some image-quality flaws previously attributed to
the quant.

Root cause of absence: the quantize recipe correctly EXCLUDES `embed_tokens` from
quantization, and the full multimodal table `model.embed_tokens.weight
[282624, 3072] BF16` (text 131125 + audio 19456 + visual 131072 + pad) ships in the
local w8a8 shards — the sidecar (rows 131125:) was simply never extracted.

- [x] Sidecar extracted on Spark from the local checkpoint (151499×3072 BF16,
      931MB) → `~/models/LongCat-Next-w8a8int8/codebook_embeddings.safetensors`
- [x] Post-fix generation captures (TTS + image) — OWNER verdicts 2026-07-31:
      image "cat looks great, no windowsill" (quality good, composition adherence
      miss); TTS onset STILL damaged post-sidecar (prod garbled the word before
      "reflection"; temp-0 started mid-word at "lection"). Sidecar fix stands on
      restore-original-behavior grounds, but it was NOT the onset root cause.
      VOCODING RULED OUT by frame arithmetic: all clips decode at a consistent
      12.5 frames/sec and wav duration == frames/12.5 exactly, so the missing
      word's frames were never accumulated — the loss is at generation/
      accumulation time (mode-entry frame drop, or the model "fading in" with
      unusable first frames). Sacrificial-leading-word probe ("Okay. Self...")
      OWNER VERDICT: damage is POSITIONAL and bounded — run b fully complete,
      run a lost only "O"/part of "Okay" (~1-2 frames); "Self reflection" intact
      in both. The 0-vs-2-frame variance argues against a fixed code off-by-N
      and toward the model's own onset (leading breath/silence frames absorb the
      loss when present), with at most a small mechanical component. Leading-
      filler prepend is a proven mitigation. NEXT: env-gated boundary
      instrumentation (log step offset of first accumulated frame + level-0
      content of first ~8 frames) to separate model fade-in from mechanical
      drop; note ~/longcat-outputs/pad_*_tts.wav suggest pad-prepend strategies
      were probed once before — check what was learned. NOTE: onset defect
      predates the rebase (v512 swallowed whole words) — NOT a rebase blocker.
- [x] Extraction script shipped (`quantize/extract_codebook_embeddings.py`) +
      README step 1 note (2026-07-31). HF model card note still pending; option:
      upload the 931MB sidecar to the HF weights repo so downloads are turnkey.
- [x] Re-examined post-fix (owner review of 4-image + 5-sentence sets, 2026-07-31):
      IMAGE geometry/adherence failure class SURVIVES the sidecar (extra ear, folded
      laptop bezel, missing person, ambiguous hands; simplest prompt scored "cat
      looks great" pre-set, "third ear" in this set — single stochastic samples, so
      no honest claim of sidecar improvement or harm on imagery; the failure class
      is simply still present). AUDIO onset unchanged in character: 2/5 sentences
      flawless, damage confined to first 0-2 phonemes, no phoneme-class correlation
      (two vowel-onset sentences went opposite ways) — stochastic head-of-clip loss.
      Sidecar retained on restore-original-mechanics grounds; it is NOT a quality
      fix for either standing defect.
      ROOT CAUSES FOUND (2026-07-31, original-vs-port seam analysis, subagent
      report; commit "Fix two boundary defects"):
      * D1 — audio onset: the port wrote multimodal special tokens RAW into the
        n-gram token table; the original NgramCache stores them as ZERO (hash
        base = text vocab 131072; specials were never hashed in training). Raw
        specials corrupt the hash context of exactly the first <=3 transcript
        tokens (12/12 embedders at t1, 8/12 t2, 4/12 t3, clean from t4) — the
        measured stochastic 0-2 phoneme onset loss. FIX: zero ids >=131072 at
        both table-write sites (overlay ngram_embedding_manager.py; env
        LCN_NGRAM_HASH_VOCAB). The kernel's ignore_tokens is NOT equivalent
        (breaks the hash window instead of zeroing through it).
      * D2 — image composition: visual token 1 was generated one step AFTER
        image_start from a zero-embedded pad, shifting the whole raster +1
        position vs training. Original generates token 1 from the image_start
        hidden in the same forward. FIX: shared _image_gen_token_step called
        from prefill detection + decode fall-through; CFG token 1 reuses the
        uncond prefill's last hidden; uncond suffix ids rebuilt explicitly
        (table now reads specials as 0).
      * Also from the report, recorded not yet acted on: transcript decoding is
        greedy vs original's sampled (D3, benign-leaning); end-of-audio differs
        (2-consecutive-confirm + flag frame dropped vs first-flag + kept, D4);
        rep-penalty window 50 frames vs all (D7); D1 partially twins into the
        visual path via the anyres resolution tokens (now fixed by D1).
      Post-fix review sets (5 TTS + 4 images) — OWNER VERDICTS (2026-07-31):
      * AUDIO: "all audio contains correct words" — the word/phoneme LOSS is
        FIXED (D1 confirmed as the onset root cause). Residual: several clips
        render the FIRST sound too fast or slightly distorted but recognizable
        ("the s in self is present but so fast I wasn't sure it was there").
        Residual is a rendering-speed/attack artifact on frame 1, not missing
        content; candidate follow-ups: D3 (transcript greedy vs original's
        sampled), first-frame vocoder attack behavior, or accept as model/
        quant baseline (BF16 reference behavior unknown).
      * IMAGES: "overall, better coherence" — windowsill PRESENT (adherence
        fixed on that prompt), bird proportions look right, market now has
        BOTH people (adherence improved). Residual geometry flaws remain
        (child's arm backwards/handless, laptop base still wrong) — the
        residual class is presumed model/quant-bound (single-sample caveat).
      D2 confirmed as a real composition-adherence defect; both boundary fixes
      VALIDATED as improvements and retained.

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

- [x] **DeepGemm accuracy flag**: RESOLVED 2026-07-31 (early, via the TTS
      investigation) — SGLANG_ENABLE_JIT_DEEPGEMM=0 vs on is bit-identical on v0.5.16
      (greedy tokens + teacher-forced logprobs): DeepGemm is inert for this w8a8_int8
      checkpoint and the `scale_fmt is not ue8m0` warning is noise. Nothing to tune.
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
