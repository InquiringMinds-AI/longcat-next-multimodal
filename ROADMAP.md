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
- [x] SUPERSEDED 2026-08-09/10 — NGRAM + generation DO compose; the crash and
      both follow-on defects are fixed (patches/spec_gen_fallback.patch, see
      research/FINDINGS.md for the full trace). The "deep allocator risk"
      estimate above was wrong in shape: no spec-shaped KV accounting had to be
      reimplemented. All three defects were ONE recurring mistake — under a
      spec-configured scheduler the SPEC bookkeeping owns the relayed
      per-request state, and plain decode prep must not also write it:
        1. input_ids  — never assigned on the spec path (the original crash).
           Cannot come from the FutureMap either: stash() early-returns for
           ngram, so output_tokens_buf is never written. Rebuilt from
           spec_info.accept_tokens the way _prepare_draft_tokens does.
        2. seq_lens   — resolve_seq_lens_cpu overwrote the plain increment, and
           since each fallback step publishes what it saw, seq_lens FROZE. The
           ngram-embedding column stopped advancing => the model generated 1369
           visual tokens against a stale context. Owner verdict on the artifact:
           "a white smudge" (vs "an apple" with the guard). Guard: skip the
           relay resolution for a flagged batch.
        3. kv_committed_len — incremented by BOTH plain prep and
           batch_result_processor (+num_accept_tokens), i.e. ~2x seq_len,
           orphaning one KV slot per decode step (measured: exactly 1405 leaked
           = the image's token count, three times). Guard: skip plain prep's
           increment for a flagged batch.
      MEASURED with both fixes, strict idle leak check ARMED: selftest 7/7,
      anthropic 5/5, degeneracy 6/6, zero leaks, agent workload 76.45 tok/s at
      100% fidelity (vs ~22 baseline). One server now does NGRAM-accelerated
      text AND image/voice generation — the speed-vs-versatility choice is gone.
      NOT YET DEFAULT: entrypoint still pairs NGRAM=>AGENT, with
      LCN_NGRAM_ALLOW_GEN=1 to opt in, pending a SOAK (25 min says nothing about
      a defect class whose signature is slow accumulation).
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
      net.
      *** CURRENT: SUPERVISED one-size-per-process run, started 2026-08-05
      10:30 UTC. Drive/monitor it with quantize/tune_moe_supervise.sh (staged at
      ~/longcat-outputs/tune_moe_supervise.sh, run detached with setsid nohup,
      log ~/longcat-outputs/supervise.log). It launches one container per batch
      size, each resuming from ~/longcat-outputs/moe_configs/checkpoints/, and
      aborts if a cycle adds no checkpoint. M=4096 was banked by run 3 before
      the switch, so it started at 1/18.
      WHY (measured, run 3): MemAvailable drifts down ~0.75GB/h WITHIN one batch
      size — 40GB at the start of M=4096, 31GB when it finished 12.5h later —
      and the boundary torch.cuda.empty_cache() reclaims NONE of it (confirmed
      flat at 31.0-31.7GB across 35 min after the boundary). A single long-lived
      process therefore reaches zero around hour 50 of a ~100h ladder, which is
      what killed run 1. That empty_cache() does nothing is the diagnostic: the
      memory is not in the CUDA caching allocator, so it is host-side, most
      likely Triton's in-process cache of compiled kernel variants (no flush
      API). Unfixable in-process -> bound the process lifetime instead.
      VERIFIED LIVE at the switch: "resume: skipping already-checkpointed batch
      sizes [4096]" and "one-size-per-run: tuning 3072, leaving 16 for later
      runs" both printed, and ckpt_M4096.json holds valid configs for both
      projections (config0 up / config1 down, USE_TMA on down). Checkpointing,
      resume, and one-size-per-run are all now proven in production, not just
      simulated.
      Low-memory alarm now at 15GB (per-size floor is ~31GB even after 12.5h).
      ETA ~87h of tuning left from 2026-08-05 10:30 UTC, plus ~2 min container
      startup x 17 cycles. ***
      RUN 3 (single-process) LAUNCHED 2026-08-04 ~16:20 CDT — restarted from scratch (run 2 was
      ~1h in) so the whole ladder runs under the resume-skip capability, which
      landed after run 2 started and could not be applied to a live script.
      Full 18-size ladder, same image/mounts/env as run 2. Startup verified the
      same way: all patch asserts passed (the "+ per-batch checkpointing" line
      prints only after the resume anchor assert too), correct target filename,
      and /proc/418/environ on `ray::BenchmarkWorker.tune` carries both
      LCN_CKPT_DIR and PYTORCH_CUDA_ALLOC_CONF. No "resume:" line, correctly —
      the checkpoint dir was empty. Checkpoint dir PROVEN to be host-backed:
      a file written inside the container was read on the host and the path
      resolves to /dev/nvme0n1p2, not overlay. Checkpoints land root-owned
      (container runs as root) — manipulate them via the container or sudo.
      Memory trend logging to ~/longcat-outputs/memtrend.csv every 5 min;
      run 2's aborted trend kept as memtrend_run2aborted.csv.
      MEASURED STARTUP CURVE (both runs agree): MemAvailable 122GB at launch ->
      79GB at 5min -> 41GB at 10min -> 40GB at 15min. ~40GB free is this
      tuner's STEADY STATE (~82GB working set: fake expert weights + 100 cached
      topk sample tensors + 20 Ray workers), reached in ~10 min. Do NOT read the
      first few samples as a baseline — a 60GB alarm set from a 4-minute sample
      false-fired immediately. Alarm now at 30GB (10GB of drift below steady
      state, 25GB of headroom above run 1's ~5GB death point). A slow decline
      away from 40GB is the signal that the allocator fix is not holding.
      (Superseded run-2 record follows.)
      RUN 2 LAUNCHED 2026-08-04 ~13:30 CDT — same full 18-size ladder, same
      image (:v0516-spec) and mounts, plus PYTORCH_CUDA_ALLOC_CONF=
      expandable_segments:True both on docker run and in-script. Startup
      verified: both patch blocks applied ("+ per-batch checkpointing" printed,
      so every anchor assert passed), correct per-channel target filename
      announced, and /proc/<pid>/environ on the live `ray::BenchmarkWorker.tune`
      worker confirms LCN_CKPT_DIR and PYTORCH_CUDA_ALLOC_CONF are both present
      in the process that actually runs the tuning. STILL UNVERIFIED until it
      runs: the checkpoint write itself (first lands when M=4096 finishes,
      ~12.5h in) and the recovery block (only executes if the tuner dies).
      Watchers armed: completion monitor + a MemAvailable<25GB creep alarm
      (run 1 died with ~5GB free), the latter so a repeat can be harvested from
      checkpoints rather than lost.
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
- [x] TUNING LADDER COMPLETE 2026-08-09 11:53 UTC — 18/18 sizes, 17 supervised
      cycles, every one exit=0. Both JSONs have 18 valid entries (M=1..4096),
      no malformed configs.
- [x] Ship configs: new_files/moe_configs/ + Dockerfile COPY into
      ${SG}/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/ (commit
      7d9e0fe). Image longcat-next-gb10:v0516-tuned. VERIFIED LOADED — the
      runtime logs, for BOTH projections:
        "Using MoE kernel config from .../triton_3_6_0/E=256,N=1024,
         device_name=NVIDIA_GB10,dtype=int8_w8a8,per_channel_quant=True[_down].json."
      GOTCHA that cost time: that line does NOT appear at startup. get_moe_configs
      is called lazily on the first MoE forward and the server runs with
      skip_server_warmup=True, so the log is silent until you send a request.
      Do not read startup silence as "config not picked up" — send one request
      first. (Calling get_moe_configs directly via docker exec also fails:
      it reads get_server_args(), which is unset outside the server process.)
- [x] PAIRED BEFORE/AFTER BENCH DONE 2026-08-09. Method is now COMMITTED
      (test/bench_decode.py, test/bench_prefill.py) — the old 21.5/22.4 numbers
      came from an uncommitted script and are NOT a safe comparator, so both
      halves were re-measured under the same script on the same machine:
      :v0516-spec (untuned, logs "Using default MoE kernel config") vs
      :v0516-tuned (logs "Using MoE kernel config from ..." for both projections).
      All-modality defaults, CUDA graphs OFF (disable_cuda_graph=True).
        decode  bs=1, 400 tok x3 prompts: 21.02 -> 21.62 tok/s = +2.9%
                (per-prompt +3.0/+2.8/+2.9%, spreads 0.1-0.3% — not noise)
        prefill 6769-token prompt x5:     2684 -> 3184 tok/s  = +18.6%
                (spreads 0.6% / 0.7%)
      WHY THE 6x GAP, and it matters for future tuning decisions: at bs=1 the
      MoE GEMM is a skinny matvec, memory-bandwidth-bound on GB10's ~270GB/s,
      so tile geometry has little room to help. Prefill runs M in the thousands
      where the same kernel is compute-bound and tile/warp/stage choices decide
      GPU utilization. TUNING PAYS WHERE THE KERNEL IS COMPUTE-BOUND.
      This retro-justifies running the FULL ladder: a decode-only tune would
      have cost hours instead of days and captured the +2.9% while missing the
      +18.6% entirely. The large-M sizes were described mid-campaign as "batch
      sizes we can't feasibly run" — that was WRONG; they are the prefill
      shapes and they run on every long prompt.
      BENCH GOTCHAS baked into the scripts: (1) prefill needs a UNIQUE NONCE
      per request or the radix cache serves it and you measure nothing —
      cached_tokens is printed to make a hit visible; (2) the warmup must be
      FULL SIZE — a short warmup leaves the large-M kernels cold and the first
      real run lands ~2.7x slow (974 vs 2684 tok/s), wrecking the median.
- [ ] *** BLOCKER 2026-08-09: THE TUNED CONFIGS BREAK THE AUDIO PATH. DO NOT
      SHIP / DO NOT PUSH new_files/moe_configs UNTIL RESOLVED. ***
      Symptom: `_assert_async_cuda_kernel: Assertion 'probability tensor
      contains either inf, nan or element < 0' failed` -> `torch.AcceleratorError:
      device-side assert triggered` -> scheduler dies, launch_server SIGKILLed,
      gateway survives so everything downstream returns connection-refused
      (those cascade failures are NOT separate bugs).
      PAIRED EVIDENCE, same image/mounts/sequence, all-modality defaults:
        :v0516-spec  (untuned) -> selftest 7/7, 0 asserts, server alive
        :v0516-tuned          -> 3 runs, 3 failures (4/7, 1/7, and the first
                                 battery), 1 assert each, server dead each time
      NOT memory: OOMKilled=false, normal headroom, memory frees only AFTER the
      process dies. NOT one bad entry: removing M=128 did not help (it made the
      run worse), and run 2 vs run 3 executed the SAME 193-token audiogen
      prefill with the SAME M=256 config, one passing and one asserting.
      => the fault is NONDETERMINISTIC. Identical config + identical shape +
      both outcomes rules out a wrong-but-deterministic tile choice; the shape
      of it is a race (likely the async-copy pipeline) that the tuned configs
      expose. Tuned configs use num_stages=5 / BLOCK_SIZE_K=256, far deeper and
      wider than get_default_config's conservative choices, on a new arch
      (sm_121). THE TUNER ONLY EVER TIMED CANDIDATES — it never compared their
      output against a reference, so a fast-but-racy config is exactly what it
      would select.
      All 3 failures were mid-size prefills (170-193 tokens) in the AUDIO path
      (audio_understanding once, audio generation twice). Large prefills (6769
      tok -> M=4096 config) and text decode never failed.
      ITERATION TOOL (big time saver): get_moe_configs honours
      SGLANG_MOE_CONFIG_DIR, so config variants can be tested by pointing the
      env at <dir>/configs/triton_3_6_0/ on the output mount — no rebuild, just
      a container restart. Variants live in ~/longcat-outputs/moe_override/.
      Crash logs preserved: ~/longcat-outputs/server_{tuned_crash2,no128,
      untuned_battery,largeM_r1,largeM_r2}.log
      ROOT CAUSE FOUND 2026-08-09 (commit d489578): USE_TMA on the DOWN
      projection produces NaN at M=1. It is set on exactly two entries, M=1 and
      M=2; excluding those two makes the whole rest of the ladder stable, so
      16 of 18 entries ship (image :v0516-tuned-16).
      Found by asking a different question — not "which config is wrong" but
      "where does the NaN first appear". LCN_NAN_CHECK=1 checks both sides of
      every MoE call and answered on the first run:
        [NAN-CHECK] layer=0 tokens=1 input_bad=False output_bad=True -> CREATED HERE
      tokens=1 is a DECODE step. THE TRAP THAT COST FOUR HYPOTHESES: every
      assert was preceded in the log by a 170-193 token audio prefill, so all of
      them chased mid-size prefills. Those prefills were bystanders — the assert
      fires at SAMPLING, after the decode steps that follow. Log adjacency is
      not causation.
      Dropping M=1/M=2 beats disabling their TMA flag: decode then resolves by
      nearest-M to the tuned M=4 config (TMA off natively), whereas disabling
      TMA in place leaves a geometry chosen BECAUSE TMA made it fast, and
      measured 20.93 tok/s — below the untuned baseline.
      SHIPPING IMAGE :v0516-tuned-16 validated: 7/7 x2, anthropic 5/5,
      degeneracy 6/6, 0 asserts, decode 21.46 / prefill 3065 tok/s.
      (Previously shipped M>=512-only, commit 31c0314: 21.37 / 3055 — the
      16-entry set is only ~0.4% better on decode; the value here is the
      diagnosis, not the throughput.)
      *** UPSTREAM-REPORTABLE: USE_TMA down-projection NaN at M=1 on sm_121 is
      an SGLang bug, not a config mistake. The tuner will pick it again on any
      Blackwell device because TMA genuinely is the fastest candidate. ***
      ⚠ NEVER benchmark with LCN_NAN_CHECK=1: .any() on both sides of 14 MoE
      layers = 28 device syncs per decoded token, pinning decode at ~20.9
      regardless of config. Two benchmark runs were wasted on that.
      Superseded: RESOLVED 2026-08-09 by shipping ONLY M>=512 (commit 31c0314). Full
      18-entry results archived in research/moe_tuning/ with the investigation
      writeup. Both refuted hypotheses are recorded there so nobody re-walks
      them: (1) single bad entry — removing M=128 made it WORSE (1/7);
      (2) pipeline depth — capping num_stages at 4 still failed (4/7 then 0/7).
      M>=512 validated across SEVEN consecutive batteries, 7/7 each, 0 asserts,
      clean output — five via SGLANG_MOE_CONFIG_DIR override, then two more on
      the actual shipping image longcat-next-gb10:v0516-tuned-safe (configs
      baked in, no override), which also passed anthropic 5/5 and degeneracy
      6/6 and measured 21.37 tok/s decode / 3055 tok/s prefill, matching the
      override runs within noise. Dropping M=1 to the 512 tile cost less decode than feared:
        untuned        21.02 tok/s decode / 2684 tok/s prefill  (stable)
        full 18-entry  21.62 (+2.9%) / 3184 (+18.6%)            (BROKEN)
        M>=512 SHIPPED 21.40 (+1.8%) / 3064 (+14.2%)            (stable)
      *** THE LESSON FOR ANY FUTURE TUNING ON THIS BOX: the sep tuner ranks
      candidates PURELY BY LATENCY and never compares their output to a
      reference, so a fast-but-racy config is exactly what it selects. Worse
      than the crashes, the capped variant returned "The\nTherl's in the image
      is a\nA red" from image_understanding and still scored PASS, because the
      test only checks for a substring. A tuning pass needs an OUTPUT-
      CORRECTNESS gate, not a battery of pass/fail modality checks. ***
- [x] CUDA GRAPHS RE-BENCHED 2026-08-09 on the tuned build — a WIN, and the
      2x2 shows tuning and graphs are ADDITIVE, not overlapping:
                        no graphs     with graphs
        untuned         21.02         21.81  (+3.8%)
        tuned           21.46 (+2.1%) 22.29  (+6.0%)
      Tuning adds ~+0.45 tok/s with or without graphs; graphs add ~+0.8 with or
      without tuning — they attack different terms (execution time vs launch
      overhead). Prefill unchanged (3065 -> 3071): prefill graphs are disabled
      on this model (incompatible with MLA attention), as expected.
      Battery 7/7, 0 asserts, 95 decode batches served from replay, and the #3
      veto machinery correctly forces eager for image/audio generation.
      MEMORY IS NOT THE OBJECTION IT WAS: capture cost 1.49GB (not the 2.79GB
      from #3) and MemAvailable after gen warmup was 9.95GB, not 4.36GB.
      => RECOMMEND making LCN_CUDAGRAPH=1 the all-modality default.
- [x] NGRAM SPEC DECODE MEASURED PROPERLY 2026-08-09 (test/bench_agent.py, new:
      verbatim source reproduction — the actual agent shape — with a FIDELITY
      check so a speedup on wrong output cannot pass):
        agent workload (warm table): 72.4 tok/s vs 22.1 baseline = 3.3x, 100% fidelity
        novel prose, COLD table:     22.9 vs 22.3 = no penalty in aggregate
            (narrative 20.8 = -6.7%, essay 22.9, technical 26.3 = +18%)
        correctness: anthropic 5/5 (incl. tool_call + tool_roundtrip), degeneracy 6/6
      The documented "7-9% novel-prose overhead" DOES NOT REPRODUCE. Two reasons:
      a 400-token generation feeds its own output into the table as it goes, so
      the drafter starts hitting partway through the SAME request; and the cost
      depends on how repetitive the content is.
      ⚠ THE N-GRAM TABLE IS SERVER-GLOBAL AND PERSISTS ACROSS REQUESTS AND
      ACROSS BENCH INVOCATIONS. Any "novel prose" measurement on a server that
      has already served similar text is measuring repetition. A cold-table
      number requires a FRESH CONTAINER and --runs 1. Two measurements were
      wasted before this was understood.
      Cold-start signature to expect: request 1 is ~baseline or slightly slower
      (empty table), then 3x once history accumulates.
      => RECOMMEND LCN_AGENT=1 imply LCN_NGRAM=1 (with an explicit off switch).
      Agent traffic is overwhelmingly repetition-shaped, correctness now checks
      out, and the cold cost is bounded to the first request.
      NOTE: NGRAM implies AGENT (generation endpoints 403), so this can never be
      a global default — only the agent profile.
- [ ] Superseded: rebench with LCN_CUDAGRAPH=1 (kernel time shrinking amplifies
      the graph win) + battery; kernel configs change scheduling not math, so temp-0
      spot check suffices unless drift appears

## 5. Performance experiments (after #3/#4 and ALL fixes — no moving targets)

### MEASURED 2026-08-09/10 — concurrency is the throughput story on this box

Aggregate throughput, tuned + graphs, unique prompts per request (prefix sharing
deliberately DEFEATED, so these are worst-case for a swarm on a shared codebase):

  conc   aggregate tok/s   per-req tok/s   vs conc=1
     1        21.2             21.2          1.00x
     4        69.8             17.4          3.30x
     8       112.5             14.1          5.32x
    16       189.2             11.8          8.94x
    32       298.1              9.3         14.09x
    64       423.6              6.6         20.02x     <- 0 failures, 28.5GB free

Concurrency 2 is PERFECTLY linear (2.00x, per-request unchanged) — the machine
is idle at bs=1. Why it bends later: at bs=1 top-12 routing touches ~12 of 256
experts, but as batch grows the union approaches ALL experts, so a forward reads
close to the full expert set. At conc=64 that is ~240GB/s against GB10's ~270 —
i.e. 423 tok/s is near the HARDWARE ceiling, not a software limit. A sparse
model behaves like a dense one at high batch.

LCN_CUDAGRAPH_BS=32 (vs default 8): +13.6% at conc=16 (166.6 -> 189.2), ~0 at
other levels, costs 0.17GB and 29s capture (bs 1,2,4,8,12,16,24,32).
=> RECOMMEND LCN_CUDAGRAPH_BS=32 alongside graphs-by-default.

### MIXED-MODALITY LOAD — what versatility actually costs (test/bench_mixed_load.py)

4 text streams held under load while generation runs concurrently:

  phase                     text/stream   retained
  baseline                    20.18          -
  + image generation           8.29         41%     image ok in 246s
  + audio generation          20.15        100%     audio ok in 17s, 192KB wav
  recovery                    19.21         95%

0 errors in any phase; server stable; MemAvailable bottoms at ~6GB with ALL
generation heads warm (above the ~3GB floor, and the tightest the system gets —
this is the number mem-fraction should be tuned against, and it argues for
LEAVING 0.72 alone).

VOICE GENERATION IS EFFECTIVELY FREE (no measurable impact on concurrent text).
IMAGE GENERATION halves text throughput for its ~4 min duration, then full
recovery. Mechanism: image gen forces EAGER via the #3 CUDA-graph veto, so text
batches lose replay and share the GPU with a long visual-token AR loop. That the
two interleave without starvation is the veto machinery working under concurrent
conditions it was designed for but never tested in.

### VERSATILITY IS THE VALUE PROP — a constraint on optimization choices

Owner ruling 2026-08-10: "nobody loads this much into ram for a one trick pony.
the value prop for this model is versatility specifically." This RULES OUT
optimizations that trade modalities for speed:
- NGRAM must stay OPT-IN, never an agent-mode default. LCN_NGRAM=1 forces
  LCN_AGENT=1, which 403s the generation endpoints — enabling it by default
  would trade away image gen + voice cloning, the capabilities that justify
  holding 92GB resident.
- CUDA graphs PASS this test: the veto keeps all 7 modalities working, verified
  under concurrent load.
- NGRAM IS SERVER-LEVEL ONLY, NOT PER-REQUEST (verified: no speculative field in
  managers/io_struct.py, sampling/sampling_params.py, or entrypoints/openai/
  protocol.py; grep sanity-checked against the same files). Structural, not an
  oversight — spec decode changes the shape of every decode batch, so the
  scheduler is either in that mode or not. Two deployment modes, switchable only
  by relaunch (~8 min): full multimodal, or text+tools at up to 3.3x with
  generation closed. Document as a MODE CHOICE, not a perf flag.


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
- [x] **Multimodal prefix-cache collision** (FIXED 2026-08-10, validated) — image/video/
      audio understanding could return ANOTHER request's media, because the processor
      pre-assigned a constant `pad_value` and thereby suppressed sglang's per-item content
      hash. Radix caching stays ON; the cache key is now correct rather than avoided.
      Proof, fix and validation in research/FINDINGS.md.
- [x] **Tool-call syntax 4** (FIXED 2026-08-10, validated) — the long-running
      "intermittent tool_calling" was never flaky: a fourth emission dialect (all args as
      one JSON object after `<longcat_arg_key>`) matched no parser branch and was dropped
      silently. `test/test_tool_parsing.py` now covers all four dialects offline.

- [ ] **Processor registration cleanup** — NOT the easy win it reads as. Investigated
      2026-08-10 against the live image (transformers **5.12.1**); do not start this
      without the findings below.

      `ProcessorMixin.from_pretrained` warns per sub-processor: *"`LongcatNextProcessor`
      defines `image_processor_class = 'Qwen2VLImageProcessor'`, which is deprecated.
      Register the correct mapping in `AutoImageProcessor` instead."* The deprecated
      branch is still fully functional in 5.12.1 — it warns, then resolves the name via
      `get_possibly_dynamic_module`. So this is future-proofing, NOT a live defect.

      **The blocker:** those hardcoded attributes are currently the ONLY thing that
      resolves the sub-processors. The checkpoint's `preprocessor_config.json` carries
      one merged config for all three modalities, with **no `image_processor_type` /
      `video_processor_type` / `feature_extractor_type` keys** and an `auto_map` that
      maps `AutoProcessor` only. Delete the attributes and `AutoImageProcessor
      .from_pretrained` has nothing to resolve from — loading breaks. So the fix is
      necessarily one of:
        (a) add `auto_map` + `*_type` entries to the checkpoint's
            `preprocessor_config.json` — but that file is part of the PUBLISHED HF
            artifact, so this is a model-repo change, not a serving-package change; or
        (b) explicit `AutoImageProcessor.register(...)` / `AutoVideoProcessor` /
            `AutoFeatureExtractor` calls in our package, run before the processor loads.

      Either path touches model loading on a production master and needs a rebuild plus
      the full battery. Cost is real; the deadline is a hypothetical future bump.

## 6b. Known limitations left UNFIXED on purpose (audited 2026-08-10)

Both were confirmed real by code reading. Neither is implemented, because the obvious fix
would be a half-fix that hides the problem rather than solving it. Recorded here so a later
session does not "discover" them and ship the shallow version.

- **No cancellation when a client disconnects.** An abandoned request keeps generating to
  completion. Verified: the request holds one of the four `_gen_slots` for its full duration.
  Note what this is NOT — the slot is always released eventually (`async with`), so this wastes
  compute, it does not leak. The reason it is not fixed: cancelling only the gateway side
  ORPHANS the backend work, which is the expensive half. A real fix needs SGLang's
  `/abort_request` wired to the disconnect, plus the request id tracked per generation. That is
  a genuine feature, not a patch, and on a single-user box the waste is bounded by the 4-slot
  cap. Worth doing if this ever serves more than one person.
- **Blocking file I/O inside async handlers.** The audio path base64-decodes and writes clips
  with plain `open()` in an `async def`. Real, and technically it stalls the event loop. Left
  alone because the payloads are seconds of audio (tens to hundreds of KB) and the stall is
  microseconds against multi-second generations — moving it to a thread pool would add
  machinery to buy nothing measurable. Revisit if long-audio input is ever supported, where the
  decode grows with clip length.

## 5b. Generation concurrency (measured 2026-08-10; premise VERIFIED 2026-08-11)

**Premise confirmed 2026-08-11** (it took three instruments; the first two gave clean wrong
answers — see FINDINGS "Cross-request head batching"). Two concurrent
`/v1/images/generations` are genuinely co-resident in one eager decode batch: 22/22 decode
batches at `#running-req: 2`, and the two requests' `[ImageGen]` progress lines land on the
SAME timestamps. So the head really is called once per request, per level, per step.

Baseline for the A/B, same box, same build (`v0516-syncfix`):

| measure | value |
|---|---|
| 2 concurrent images, wall | **410.8 s** |
| generation steady state, n=2 | 58 s / 10 raster rows |
| generation steady state, solo | ~36.5 s / 10 raster rows |
| => n=2 costs | **1.59x** solo for 2x the work |

- [ ] **NEW, possibly bigger than everything below: the REFINER is serial and is ~47% of n=2 wall
      time.** Generation of both images finished at 08:04:38; the two refiner passes then ran ONE
      AT A TIME (~98 s each; req=25 saved 08:06:16, req=26 only after). 214 s generating vs ~196 s
      refining. Nothing has ever profiled or batched `image_refiner.py`. Measure before assuming.

- [ ] **int8 the generation heads — HIGHEST VALUE, and the only item that touches single-image
      latency.** They are 71/71 BF16 (audio 2.86GB, visual 1.76GB) while the backbone is int8;
      they are read 8x per frame. On a box where "its all ram bandwidth choking us out" (owner),
      halving their width halves the traffic of every generation. Needs owner eyes/ears on
      output — this changes what the images and voice actually are.
- [ ] **Micro-benchmark the depth head, sweeping BOTH batch (1/2/4/8) AND dtype (bf16/int8),
      WITH THE MODEL UNLOADED.** Sizes both items above before either is built. Needs
      1.2-2.4GB, so never against a live server at ~117GB in use.
- [x] **BUILT 2026-08-11 — batch `CasualDepthTransformerHead` across concurrent generating
      requests** (was: one batch-1 call per request per level per step). `_image_gen_decode_step`
      now collects the requests needing a visual token into `pending` and `_image_gen_flush`
      issues ONE head call for the group; falls back to the per-request path when CFG is active
      (CFG fuses its rows pairwise, so the batch axis is already spoken for). Image path only —
      audio deferred.
      Equivalence verified where it CAN be, at the logits (`test/test_head_batching.py`, 40/40):
      batching changes RNG order so output images are legitimately different and cannot be
      diffed; row logits vs batch-1 agree to ~4e-7 with matching argmax, and batch-COMPOSITION
      independence is exactly 0.00e+00. Still needs owner eyes on output, per the standing rule.
      Does NOTHING at n=1 — it is a concurrency optimization.
      ⚠ **The first version of this SHIPPED A CORRECTNESS BUG and its speedup numbers were
      measured on contaminated output.** With exactly 2 concurrent requests both images came
      out the SAME — the owner caught it by LOOKING ("barn a, barn b, sailboat, barn b
      duplicate") after a fully green battery. Cause: `_generate_image_codebook_step` gated
      its CFG fusion on `logits.shape[0] == 2`, using batch size as a proxy for "these rows
      are a cond/uncond pair"; batching gave 2 a second meaning, so request B's logits were
      fused into request A's and A's sampled token broadcast to both rows. FIXED by gating on
      `uncond_hidden is not None` + explicit shape guards; regression test
      `test/test_codebook_batching.py` (5/7 broken → 7/7 fixed, offline/CPU/deterministic).
      Isolated with `LCN_HEAD_BATCH=0`, which forces the per-request path in the same build:
      taxi request returned a barn with batching ON, a taxi with it OFF.
      **Speed, same build, flag off vs on: 414 s → 327 s (−21%) for 2 concurrent images.**
      Re-validation of the FIXED build is the open item — no number here is trustworthy until
      the paired output has been looked at again.
- [ ] **Audio head batching is NOT the same change — do not copy the image one.** Assessed
      2026-08-11 by reading `_generate_audio_codebook_step`: sampling there carries PER-REQUEST
      state that the image path does not have. `prev_ids` (the last 50 frames) feeds a
      repetition penalty via `_sample_codebook_logits(logits, level, prev_ids)`, frame counts
      differ per request so a batched `prev_ids` needs padding/masking, and `next_token_ids`
      is allocated `torch.zeros(1, num_codebooks)`. Real correctness risk, and TTS's bigger
      usability win is streaming (Tier 3) rather than concurrent throughput.
- [ ] Lower priority than it looks: KV-cache the head across levels. Saves O(depth^2) recompute
      but not weight reads; if the head is bandwidth-bound this is nearly free of benefit.
- Measured baseline: n=1 238.8s, n=2 422.9s (1.77x), n=4 796.9s (3.34x) — ~84% serial.
- NOT fixable this way: single-image latency. 8 sequential weight-read passes per frame x 1369
  frames, levels sequentially dependent by construction.

## 7. Final numbers + legibility (last — documents whatever the above produced)

- [ ] Re-run the full bench suite on the final configuration; update README numbers
- [ ] README: lead with the Claude Code / agent-mode headline (currently mid-file)
- [ ] HF model card: agent mode + Anthropic route + updated serving guidance
- [ ] Short terminal recording of Claude Code driving the container
