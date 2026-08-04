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
- [x] Port overlay to v0.5.16, full revalidation — **SHIPPED 2026-07-31**: merged to
      master, pushed, image promoted to :latest on Spark (:v0512-final = rollback
      tag). Owner verdict: "a genuine improvement." Final battery on the shipped
      build: 7/7 selftest, 5/5 anthropic, 6/6 degeneracy. The investigation that
      unblocked it also root-caused and fixed two day-one generation defects (D1
      onset, D2 raster start — see below) and restored the codebook sidecar.
      History of the blocker (kept for the record): port state: three-way merges done; eos_token_id
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
        content. D3 (sampled transcript, original semantics) TESTED 2026-07-31:
        owner verdict "most are good" but no clear onset improvement — failure
        texture changed (junk-syllable prepend "uf...Self", vowel substitution
        "ee-very") at a similar rate (2/5 vs 3/5; n=5 = noise, no ranking
        claim). D3 KEPT on faithfulness grounds (LCN_TRANSCRIPT_GREEDY=1
        reverts). Residual first-frame variance now presumed MODEL BASELINE
        (all known mechanical divergences fixed; failures are wrong-content
        picks, not missing content).
        Text lead-in A/B (owner ears, n=1 each, 2026-07-31): dash lead ("— ")
        rendered silent and delivered an INTACT onset; "Mm." absorbed the damage but INVITED ELABORATION —
        1-2 words' duration of nonsense before intelligible speech (8.86s clip); ellipsis was HARMFUL (content
        substitution: "Self reflection"->"research" — damage landed inside
        the first word). Confirms the absorber theory; silence-FRAME injection
        (LCN_TTS_SILENCE_FRAMES, engine-level deterministic absorber) is the
        favored mechanism.
        N=1 + TRIM OWNER VERDICT (2026-07-31): "these all work" — zero garble,
        all female, natural; residual = accent/tone variation ("plausibly the
        same speaker, but if they were doing impressions") — generic-silence
        anchoring still dilutes the clone slightly.
        REFERENCE-VOICE LEAD: TESTED AND REJECTED (owner: "all worse" — identity
        NOT maintained, one clip devolved into nonsense). Post-mortem: a
        mid-utterance reference frame, even the quietest, implies DISCONTINUITY
        as generation history — worse than neutral silence. Code reverted.
        SHIPPING DEFAULT (baked into entrypoint): LCN_TTS_SILENCE_FRAMES=1 +
        LCN_TTS_TRIM_LEAD_MS=150 (the owner-validated "these all work" config).
        Residual impressions-grade identity wander = accepted baseline for now;
        remaining ideas if ever revisited: longer/denser reference conditioning,
        temperature reduction on the first free frames.
        INJECTION N=2 OWNER VERDICT (2026-07-31): "the utterances are perfect"
        — ALL FIVE sentences intact, onset defect fully absorbed. One issue:
        the model EXTENDS injected silence by momentum (20-40% of each clip was
        silent lead). SECOND ISSUE (owner, hedged "I think"): SPEAKER IDENTITY
        drifted in the N=2 set — male AND female voices where every prior set
        held one consistent female clone. Plausible mechanism: injected frames
        are encoded DIGITAL silence = speaker-neutral history at exactly the
        identity-anchoring early frames; the onset garble may partly have BEEN
        the anchoring process. Ladder: (a) N=1 + trim in test (doubles as the
        identity test); (b) if drift persists -> dash-lead + trim (model-
        GENERATED silence is identity-anchored by construction, dash onset was
        intact); (c) speaker-colored lead frames (encode a known-good clip's
        own opening) as the injection variant.
      * IMAGES: "overall, better coherence" — windowsill PRESENT (adherence
        fixed on that prompt), bird proportions look right, market now has
        BOTH people (adherence improved). Residual geometry flaws remain
        (child's arm backwards/handless, laptop base still wrong) — the
        residual class is presumed model/quant-bound (single-sample caveat).
      D2 confirmed as a real composition-adherence defect; both boundary fixes
      VALIDATED as improvements and retained.
      Baseline calibration (owner, 2026-07-31): the ORIGINAL stand-up samples
      were never perfect either — his acceptance bar then was "recognizable
      subjects rather than abstract art." D2 was in the serving path from day
      one, so no prior era had clean raster initiation; the current post-fix
      state is plausibly this deployment's best-ever image quality, and the
      residual geometry class has existed since stand-up (consistent with
      model/quant bound).

## 2. Incremental streaming (orthogonal — gateway only, no relaunch risk)

Both the OpenAI tool path and the Anthropic route currently buffer the whole completion
(~20 s of dead air for a 400-token answer at ~21 tok/s). Stream tokens through live;
start buffering only when `<longcat_tool_call>` appears mid-stream; emit parsed tool
calls at the end. Applies to `/v1/chat/completions` (tools branch) and `/v1/messages`.
No interaction with any engine work below.

- [x] OpenAI route: stream-with-tool-detection — VALIDATED 2026-07-31: first
      content delta at 0.52s of a 4.23s completion (was: full buffer wait), 81
      live deltas; streamed tool call emits tool_calls deltas +
      finish_reason=tool_calls correctly.
- [x] Anthropic route: real SSE deltas — VALIDATED: 96 text deltas on a
      multi-sentence answer (buffered path emitted exactly 1); 5/5 route checks.
- [x] test_anthropic streaming check tightened (>=3 text deltas required)
      Implementation: stream_tools.ToolStreamFilter (shared, unit-tested) —
      rolling-tail withholding so partial markers never leak; silent-buffer
      from first marker; marker-without-calls releases the swallowed text.

## 3. n-gram embedding rework (engine-deep — the double unlock)

Both abandoned speed levers failed in the same file: CUDA graph capture and NGRAM
speculative decoding each die in `n_gram_embedding.py` forward, which contains
`if ignored_mask.any():` — a host-side sync + dynamic branch in the decode path
(capture poison) and history-hash indexing that draft positions violate (illegal
memory access). Make the layer branch-free and draft-position-safe.

2026-07-31 root-cause pass (v0.5.16 base — much of the old diagnosis is stale on
this base; upstream now ships NgramEmbeddingInfo buffers wired into the decode
CUDA-graph runner, including NGRAM-verify graph capture):
* CAPTURE poisons were OURS, not the kernel's: (a) the overlay layer's
  `if ignored_mask.any():` host sync (base v0.5.16 forward is branch-free);
  (b) the mm model's Step-3 state machines running per-element `.item()`
  loops on EVERY decode — also a 2-sync/token latency tax on plain text.
* NGRAM illegal access root-caused from source: TARGET_VERIFY is_extend()=True,
  so `_init_ngram_embedding_info` read extend_prefix_lens/extend_seq_lens,
  which hand-built verify batches never populate → NgramEmbeddingInfo.create
  leaves column_starts UNINITIALIZED (torch.empty) → OOB table indexing.
  (Garbage token VALUES can't crash — the hash is %-bounded; garbage COLUMNS can.)
* The hash kernel reads context ONLY from its token table (its `tokens` arg is
  dead code), so spec drafts must be table-written before a verify forward.
  Drafts are a TREE; the kernel walks linearly → chain drafts only
  (bfs breadth pinned to 1) makes linear writes semantically exact.
* CUDA graph replay skips the model's Python forward entirely → any batch that
  might need the mm state machines must veto replay and run eager.

Implementation (all in-tree; image :v0516-graph):
- [x] Layer: branch-free ignored-mask (`torch.where`, no host sync)
- [x] Gen-trigger latch: post-sample on-GPU scan of sampled ids for the two
      gen-ENTRY tokens (audiogen_start/image_start) + async 1-byte flag to
      pinned host memory, folded into a STICKY latch read at the next forward
      (= the step the trigger arrives as input — one step late is exactly on
      time). Latch consumed on observation; 64-step decay guard for triggers
      whose request died. Step-3 loops now gated on
      (gen states non-empty | latch) → ZERO host syncs on steady-state text
      decode. Scan hooks: ngram manager update_after_decode (normal path) +
      spec worker accept path.
- [x] patches/decode_graph_gen_veto.patch: can_run_graph consults
      model.lcn_cuda_graph_veto() (gen active or trigger pending → eager)
- [x] patches/ngram_spec_verify.patch: TARGET_VERIFY branch in
      _init_ngram_embedding_info (column_starts=seq_lens, req_lens=
      draft_token_num); NGRAM worker writes drafts pre-verify + accepted
      tokens post-verify via lcn_write_spec_tokens (same specials-to-zero
      hashing rule); gen-active rounds fall back to plain decode (same veto)
- [x] entrypoint: LCN_CUDAGRAPH=1 now also pins --disable-prefill-cuda-graph
      (mm prefill forward is host-driven, not capture-safe); LCN_NGRAM=1 pins
      bfs breadth 1 (chain drafts)
Results (2026-07-31, image :v0516-spec = both phases):
- [x] CUDA graph capture WORKS: bs [1,2,4,8], 34s capture, +2.79GB. Full
      battery GREEN with graphs on (7/7, 5/5, 6/6) — gen artifacts produced,
      proving the veto→eager path fires (log split: 124 graphed / 63 eager
      batches, no decay warnings). MemAvailable after gen warmup 4.36GB
      (was ~5GB pre-graphs; thinner but above the ~3GB floor). Decode bench
      bs=1: 22.4 tok/s graphs vs 21.5 eager = +4.2% (MoE-GEMM-bound as
      expected; the win compounds after #4 tuning).
- [x] GRAPH REPLAY IS BIT-IDENTICAL TO EAGER (temp-0, 3 prompts) — stronger
      than expected; no graph-numerics caveat needed.
- [x] NGRAM spec decode NO LONGER CRASHES; verbatim-repetition probe: correct
      output at 39.6 tok/s (+84% vs 21.5; implies ~1.8x mean accept). Novel
      prose: accept len ~1.0-1.4, ~7-9% overhead — canonical prompt-lookup
      shape; keep opt-in, document as agent/repetitive-workload lever.
- [x] Temp-0 identity gate: REFRAMED — this engine's temp-0 output is
      RADIX-STATE-DEPENDENT (same build, same prompt: warm-vs-cold radix
      flips prompt 1 at ~char 170; warm repeats are stable). The pre-rework
      baseline was captured on a warm long-lived container, so cross-build
      bitwise identity is unattainable by that comparator; 2/3 prompts
      matched anyway. Spec-vs-nospec temp-0 also diverges (verify runs the
      prefill attention kernel — different numerics, same near-tie flips);
      accept-rate + verbatim-repetition correctness is the discriminating
      instrument for hash-geometry correctness, and it passes.
- [x] AIRTIGHT rework gate PASSED: cold first-run temp-0 capture from the
      pre-rework :latest image is BIT-IDENTICAL (3/3 prompts) to the rework
      build's cold first-run. The rework is output-preserving; every earlier
      divergence was radix-state, as diagnosed.
- [x] NGRAM + generation DO NOT COMPOSE — design changed to NGRAM⇒AGENT:
      the first attempt (spec worker falls back to plain decode while gen
      active) CRASHED the engine on the first gen decode round — under a
      spec-configured scheduler, decode batches arrive WITHOUT input_ids
      (the verify rewrite is what sets them from draft tokens), so the
      skipped rewrite left a half-built batch (registry fill: positions
      dst=(0,) src=(1,); named via temporary fill_from instrumentation).
      Fixing that means reimplementing spec-shaped KV-slot accounting for
      plain rounds — deep allocator risk for a combination with no real
      deployment: gen serving doesn't want spec, agent serving 403s gen.
      SHIPPED DESIGN: LCN_NGRAM=1 implies LCN_AGENT=1 (entrypoint pairs +
      loud log); the model itself disables the gen machinery under
      LCN_AGENT=1 (no state entry from prefill markers, no trigger latch
      work) — closes a pre-existing leak where raw gen markers in a chat
      prompt could strand a gen state on a reused req_pool_idx. The
      worker patch keeps only the draft/accept table writes.
- [x] Battery on the NGRAM⇒agent config: degeneracy 6/6, anthropic 5/5
      (incl. tool_call + roundtrip), gen endpoints 403 as designed, audio
      UNDERSTANDING passes; selftest image/video-understanding "failures"
      are a cascade (they reuse the 403'd generated image — bare
      `assert img_b64`), not real. Repetition probe: 46.9 tok/s.
      ONE REAL OBSERVATION: the selftest tool_calling probe missed once
      (model answered in prose, no tool call) on the cold corpus; manual
      re-probe 6/6 OK + anthropic tool checks green. Spec decode makes
      temp-0 output corpus-state-dependent, so occasional near-tie flips
      are expected-class; at n≈9 the miss rate is not distinguishable
      from rare-but-real. Documented as a caveat on the opt-in flag —
      selftest under NGRAM needs an agent-mode variant if this mode
      graduates.
- [x] Owner eyes/ears on gen review sets (2026-07-31): "images are good,
      maybe better. they still have defects though. the audio is good too."
      — generation paths confirmed non-regressed (residual geometry defect
      class persists, the known model/quant-bound baseline). GATE PASSED.
- [ ] Decide ship defaults (likely: both features stay opt-in env gates;
      LCN_AGENT deployments are the natural place for LCN_CUDAGRAPH=1)

## 4. MoE kernel tuning (after #3 — tune the final decode path once)

Every launch warns: default fused-MoE Triton config for
`E=256,N=1024,device_name=NVIDIA_GB10,dtype=int8_w8a8,per_channel_quant=True` (and the
`_down` variant) is missing. Decode (~21 tok/s bf16) is MoE-GEMM-bound. Run a targeted
sweep for exactly these shapes and bake the JSONs into the image. Done after #3 so the
tuned path is the shipping path (config files are also keyed by triton version — another
reason #1 settles first).

- [ ] Targeted `tuning_fused_moe_triton_sep.py` sweep (both up and down proj; artifacts
      on a mounted volume, never in an `--rm` container)
      *** RUN 1 FAILED 2026-08-04 — TOTAL LOSS, 4 days, zero configs written. ***
      14 of 18 batch sizes had completed (4096..16) and it died during the last
      small-M sizes: `ray.exceptions.OutOfMemoryError` — Ray's memory monitor
      OOM-killed the BenchmarkWorker at 116.84GB/121.69GB node usage (95%
      threshold), `ray.get(outputs)` re-raised in `_distribute`, main() never
      reached save_configs_sep. Verified unrecoverable: no JSON in the container
      bench dir, no Ray spill, nothing on the mount, and the tuner never prints
      configs. ROOT CAUSE (strong hypothesis, not proven): the container was
      launched with `--entrypoint bash`, which bypasses entrypoint.sh, so
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True was never set —
      docker inspect confirmed it absent from the container env. On GB10 unified
      memory the caching allocator's per-shape blocks count against system RAM
      and accumulate across ~30k tuned configs. Both fixes are now in
      quantize/tune_moe_gb10.sh (allocator setting exported in-script;
      torch.cuda.empty_cache() per batch size; plus the checkpointing below, which
      would have preserved 14/18 sizes had it been active). Do NOT "fix" this by
      raising RAY_memory_usage_threshold — 116GB is already inside the
      ~110-115GB band where Spark hard-powers-off; the OOM monitor is the safety
      net. Run 2 scope (full ladder vs decode-only M) is an owner decision.
      Historical detail of run 1 follows.
      RUN 1 ran since 2026-07-31
      ~14:36 in container `lcn-moe-tune` on Spark (image :v0516-spec, entrypoint
      quantize/tune_moe_gb10.sh staged at ~/longcat-outputs/tune_moe_gb10.sh; topk
      captures in ~/longcat-outputs/topk_ids, 14 MoE layers x2 from a 16k-token
      diverse doc+code prompt). OWNER DECISION 2026-07-31: run the FULL 18-batch-size
      ladder to completion (~2.5-3 days measured — M=4096 alone ~19h at 36s/config;
      the tuner walks batch sizes sequentially, largest first) rather than rescope to
      decode-only M; he'll work on non-Spark things meanwhile. Serving is DOWN for
      the duration (tuner owns the GPU; Yuki stack also down). The tuner is
      SESSION-INDEPENDENT: if the CC session dies, check `docker logs lcn-moe-tune`
      and ~/longcat-outputs/moe_configs/ for the two JSONs
      (E=256,N=1024,device_name=NVIDIA_GB10,dtype=int8_w8a8,per_channel_quant=True
      .json + _down). Runtime engine loads them from
      configs/triton_3_6_0/ under the moe_runner/triton_utils dir (or
      SGLANG_MOE_CONFIG_DIR env).
      MEASURED per-batch-size wall times from run 1 (M: hours) 4096:12.5,
      3072:10.4, 2048:9.1, 1536:8.6, 1024:8.0, 512:7.5, 256:6.9, 128:6.1,
      96:5.7, 64:5.1, 48:4.7, 32:4.0, 24:3.5, 16:3.5, 8:~0.9. Per-config cost is
      dominated by Triton JIT + Ray dispatch, not GEMM work, so the decay
      flattens at ~3.5h through M=16; the cliff only arrives at M<=8, where most
      of the 1920 candidates are invalid for the shape and skip via
      OutOfResources without compiling. Total for the full ladder ~4.2 days.
      Budget a decode-only rerun (M<=48) at roughly 17h, not hours.
      CHECKPOINTING (added 2026-08-03, NOT active in the current run — the
      staged copy on Spark was deliberately left untouched because bash reads
      a running script by byte offset): the stock tuner buffers all 18 results
      in the Ray driver and calls save_configs_sep ONCE at the end, and never
      prints the configs, so a crash loses the whole run with nothing
      recoverable from the log. quantize/tune_moe_gb10.sh now patches
      BenchmarkWorker.tune to dump each batch size to
      $OUT/checkpoints/ckpt_M<N>.json as it completes, and a post-run recovery
      block rebuilds the two final JSONs from those checkpoints (same filename
      derivation, sort_config, ascending-M order as save_configs_sep) if the
      tuner died first. Anchor verified unique against the real source; the
      sep-tuner patch section now self-skips when already applied. A partial
      ladder is still usable — the runtime interpolates across the M values
      present.
- [ ] Ship configs: repo new_files/moe_configs/ + Dockerfile COPY into
      ${SG}/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/; verify launch
      log says "Using MoE kernel config from ..." (today's builds warn it's missing)
- [ ] Before/after decode bench (baselines: 21.5 tok/s eager / 22.4 graphs, 400-token
      essay probe x3) + rebench with LCN_CUDAGRAPH=1 (kernel time shrinking amplifies
      the graph win) + battery; kernel configs change scheduling not math, so temp-0
      spot check suffices unless drift appears

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
