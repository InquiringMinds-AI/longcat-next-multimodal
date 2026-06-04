#!/usr/bin/env python3
"""
PHASE A — LongCat-Next standalone IMAGE-GENERATION backbone loop (cond-only, no CFG).

Runs INSIDE the lc-vision container image with the ~/lc_overlay/* mounts.
Drives a standalone sglang ModelRunner over the LongCat backbone:
  prefill prompt -> capture LAST hidden -> 8-codebook depth-head inner loop
  -> sample -> A1 re-embed via VisualEmbeddingBridge -> next EXTEND/DECODE step.
Accumulates [324,8] raw codebook ids -> /tmp/gen_ids.pt, then EXITS.

KEY ARCHITECTURE FACT discovered during recon:
  The NVFP4-bf16mla checkpoint's model.embed_tokens.weight was TRUNCATED to
  [131125,3072] (text+special only). The generation-side VisualEmbeddingBridge
  embedding_layers are sliced from embed_tokens at visual_offset=150581, which
  only exist in the ORIGINAL bf16 checkpoint /models/LongCat-Next
  (embed_tokens [282624,3072]). So we load the bridge embeddings + transformer_block
  from the ORIGINAL checkpoint, while the backbone runs on NVFP4. visual_head.* and
  pre_buffer.* exist in both; we load them from NVFP4 (served model) for consistency.
"""
import os, sys, json, glob, struct, gc, traceback
import numpy as np
import torch

NVFP4 = os.environ.get("MODEL_PATH", "/models/output/LongCat-Next-NVFP4-bf16mla")
ORIG  = "/models/LongCat-Next"

# Token ids
IMG_START = 131106
IMG_END   = 131107
IMG_PAD   = 131108
IMG_NEWLINE = 131109
VISUAL_OFFSET = 150581
HIDDEN = 3072
NUM_CB = 8
CB = 16384            # per-codebook size (id range 0..16383); slot 16384 is EOS -> masked
TOKENS_H = TOKENS_W = 18
RMS_EPS = 1e-5

_SUFFIX = "<longcat_img_token_size>18 18</longcat_img_token_size><longcat_img_start>"
_GEN_PROMPT = os.environ.get("GEN_PROMPT",
    "A single large red circle centered on a plain white background.")
PROMPT = _GEN_PROMPT.rstrip() + " " + _SUFFIX  # canonical: " ".join(question.split()) keeps ONE space before <longcat_img_token_size>

# Sampling for depth head (per task spec)
TEMP = float(os.environ.get("DEPTH_TEMP", "0.5"))
TOP_P = float(os.environ.get("DEPTH_TOP_P", "0.75"))
TOP_K = int(os.environ.get("DEPTH_TOP_K", "1024"))
TOP_K_LEVELS = None  # optional per-level [8] top_k override (set by driver)

CODEBOOK_BASE = 131125            # text_vocab_plus_multimodal_special_token_size
# visual_offset_vals = cumsum([150581] + [16384]*7)  -> STRIDE 16384 (NOT 16385).
VISUAL_OFFSET_VALS = [VISUAL_OFFSET + i * CB for i in range(NUM_CB)]  # [150581,166965,...,265269]

# FIX 2: classifier-free guidance
USE_CFG = True
CFG_SCALE = float(os.environ.get("CFG_SCALE", "1.8"))
PROMPT_COND = PROMPT
PROMPT_UNCOND = ("<longcat_img_token_size>18 18</longcat_img_token_size><longcat_img_start>")


def log(*a):
    print(*a, flush=True)


# ---------- raw safetensors slice reader (no full-tensor load where avoidable) ----------
def _index(path):
    idxf = glob.glob(os.path.join(path, "*index*.json"))[0]
    return json.load(open(idxf))["weight_map"], path


def _resolve(path, rel):
    cand = os.path.join(path, rel)
    if os.path.exists(cand):
        return os.path.realpath(cand)
    # NVFP4-bf16mla index may reference sibling NVFP4 base shards
    return os.path.join("/models/output/LongCat-Next-NVFP4", os.path.basename(rel))


def load_tensor(idx, path, key):
    from safetensors import safe_open
    sf = _resolve(path, idx[key])
    with safe_open(sf, framework="pt") as f:
        return f.get_tensor(key)


# ============================================================================
# 1. Build the generation-side modules (bridge + image head) — host-side, CPU then cuda
# ============================================================================
def build_gen_modules():
    log("[A.modules] building VisualEmbeddingBridge + OmniImageHead ...")
    sys.path.insert(0, os.path.expanduser("~/Projects/LongCat-Next-inference"))
    from modules.visual_emb import VisualEmbeddingBridge
    from modules import image_head as _ih
    from modules.image_head import OmniImageHead

    # Container's flash_attn is a hollow stub -> flash_attn_varlen_func is None.
    # Install a pure-PyTorch SDPA varlen fallback (matches longcat_next_heads.py).
    if _ih.flash_attn_varlen_func is None:
        import torch.nn.functional as _F

        def _sdpa_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal=False, window_size=(-1, -1), **kw):
            outs = []
            cq = cu_q.tolist(); ck = cu_k.tolist()
            for i in range(len(cq) - 1):
                qi = q[cq[i]:cq[i + 1]].unsqueeze(0).transpose(1, 2)  # [1,H,sq,D]
                ki = k[ck[i]:ck[i + 1]].unsqueeze(0).transpose(1, 2)
                vi = v[ck[i]:ck[i + 1]].unsqueeze(0).transpose(1, 2)
                oi = _F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal)
                outs.append(oi.transpose(1, 2).squeeze(0))
            return torch.cat(outs, dim=0)

        _ih.flash_attn_varlen_func = _sdpa_varlen
        log("[A.modules]   installed SDPA varlen fallback in image_head")

    codebook_sizes = [CB] * NUM_CB

    # --- bridge: embedding_layers sliced from ORIGINAL embed_tokens, transformer_block from pre_buffer
    cfg = json.load(open(os.path.join(NVFP4, "config.json")))
    vc = cfg["visual_config"]
    bridge = VisualEmbeddingBridge(
        codebook_sizes=codebook_sizes,
        hidden_size=HIDDEN,
        intermediate_size=cfg["intermediate_size"],   # 6144 (matches pre_buffer? checked below)
        hidden_act=cfg.get("hidden_act") or "silu",
        rms_norm_eps=RMS_EPS,
    )
    # bridge embeddings from ORIGINAL embed_tokens rows [offset:offset+CB+1] per level
    oidx, opath = _index(ORIG)
    log("[A.modules] loading ORIGINAL embed_tokens (for bridge embedding slices) ...")
    et = load_tensor(oidx, opath, "model.embed_tokens.weight")   # [282624,3072] cpu bf16
    log(f"[A.modules]   orig embed_tokens {tuple(et.shape)} {et.dtype}")
    emb_sd = {}
    offset = VISUAL_OFFSET
    for i in range(NUM_CB):
        emb_sd[f"{i}.weight"] = et[offset:offset + CB + 1, :].clone()
        offset += CB
    bridge.embedding_layers.load_state_dict(emb_sd, strict=True)
    del et; gc.collect()

    # transformer_block (DecoderLayer) <- pre_buffer (mlp + pre_layernorm)
    nidx, npath = _index(NVFP4)
    pb = "model.visual_tokenizer.visual_embedding_layer.pre_buffer."
    tb_sd = {
        "mlp.gate_proj.weight": load_tensor(nidx, npath, pb + "mlp.gate_proj.weight"),
        "mlp.up_proj.weight":   load_tensor(nidx, npath, pb + "mlp.up_proj.weight"),
        "mlp.down_proj.weight": load_tensor(nidx, npath, pb + "mlp.down_proj.weight"),
        "pre_layernorm.weight": load_tensor(nidx, npath, pb + "pre_layernorm.weight"),
        "pre_layernorm.bias":   load_tensor(nidx, npath, pb + "pre_layernorm.bias"),
    }
    # NOTE: bridge MLP intermediate must match pre_buffer (8192). Rebuild MLP if mismatch.
    pb_inter = tb_sd["mlp.gate_proj.weight"].shape[0]
    if bridge.transformer_block.mlp.gate_proj.weight.shape[0] != pb_inter:
        log(f"[A.modules]   rebuilding bridge MLP intermediate {bridge.transformer_block.mlp.gate_proj.weight.shape[0]} -> {pb_inter}")
        from modules.visual_emb import MLP
        bridge.transformer_block.mlp = MLP(HIDDEN, pb_inter, cfg.get("hidden_act") or "silu")
    bridge.transformer_block.load_state_dict(tb_sd, strict=True)
    bridge = bridge.to("cuda").to(torch.bfloat16).eval()
    log("[A.modules]   bridge ready")

    # --- image head (OmniImageHead) from NVFP4 visual_head.*
    head = OmniImageHead(
        hidden_size=HIDDEN,
        codebook_sizes=codebook_sizes,
        image_head_transformer_ffn_scale=vc["image_head_transformer_ffn_scale"],
        image_head_transformer_dims=vc["image_head_transformer_dims"],
        image_head_transformer_layers=vc["image_head_transformer_layers"],
        image_head_enable=False,
    )
    head_sd = {}
    for k in nidx:
        if k.startswith("visual_head."):
            head_sd[k[len("visual_head."):]] = load_tensor(nidx, npath, k)
    missing = head.load_state_dict(head_sd, strict=False)
    log(f"[A.modules]   image head loaded; missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")
    if missing.missing_keys:
        log("    head missing:", missing.missing_keys[:10])
    head = head.to("cuda").to(torch.bfloat16).eval()

    # --- FIX 1: load mm_embed_rows (== original embed_tokens[131125:282624]) so the
    # generation feedback can reuse the EXACT understanding embed mechanism
    # (_embed_visual_ids), rather than a parallel reconstruction.
    from safetensors import safe_open as _so
    mm_path = "/models/lc_mm_embed_rows.safetensors"
    with _so(mm_path, framework="pt") as f:
        mm_embed_rows = f.get_tensor("mm_embed_rows").to("cuda").to(torch.bfloat16)
    log(f"[A.modules]   loaded mm_embed_rows {tuple(mm_embed_rows.shape)} from {mm_path}")
    return bridge, head, codebook_sizes, mm_embed_rows


# ============================================================================
# FIX 1 — feedback embedding IDENTICAL to the proven understanding path.
#   understanding get_image_feature(one position):
#     offset_id[i] = raw_id[i] + visual_offset_vals[i]
#     summed = sum_i mm_embed_rows[offset_id[i] - 131125]
#     feat   = visual_embedding_layer(summed)   # == bridge.transformer_block (pre_buffer)
# ============================================================================
def embed_visual_ids_understanding(raw_ids, mm_embed_rows):
    """raw_ids: [N, NUM_CB] raw codebook ids (0..16383). Returns [N, HIDDEN] summed
    pre-pre_buffer embedding, mirroring longcat_flash._embed_visual_ids exactly."""
    summed = None
    for i in range(NUM_CB):
        offset_id = raw_ids[:, i].long() + VISUAL_OFFSET_VALS[i]
        idx = (offset_id - CODEBOOK_BASE).clamp(min=0, max=mm_embed_rows.shape[0] - 1)
        e = mm_embed_rows[idx]
        summed = e if summed is None else summed + e
    return summed   # [N, HIDDEN]


def feedback_embed(raw_ids, bridge, mm_embed_rows):
    """Generation feedback for one image position's 8 raw codebook ids, computed via
    the proven understanding mechanism: summed mm-embed -> pre_buffer transformer_block.
    raw_ids: [1, NUM_CB]. Returns [1, HIDDEN] bf16."""
    summed = embed_visual_ids_understanding(raw_ids, mm_embed_rows)  # [1, HIDDEN]
    feat = bridge.transformer_block(summed.to(torch.bfloat16))[0]
    return feat.reshape(1, HIDDEN).to(torch.bfloat16)


def verify_feedback_match(bridge, mm_embed_rows):
    """Encode a real image, get raw VQ ids, then run them through BOTH (a) the old
    VisualEmbeddingBridge feedback and (b) the new understanding-path feedback. Report
    whether they are identical (isolates whether the old feedback was the content bug).
    Memory-safe: tries to encode /tmp/red_rect.png; if unavailable, synthesizes random
    raw ids (the embed paths are id-driven, so the equality check is still valid)."""
    raw = None
    try:
        import os as _os
        if _os.path.exists("/tmp/red_ids.pt"):
            raw = torch.load("/tmp/red_ids.pt").to("cuda")
            log(f"[A.verify] using cached encoded ids /tmp/red_ids.pt {tuple(raw.shape)}")
    except Exception as e:
        log(f"[A.verify] cached id load failed: {e}")
    if raw is None:
        # synthesize a small batch of valid raw ids to exercise both embed paths
        torch.manual_seed(0)
        raw = torch.randint(0, CB, (16, NUM_CB), device="cuda")
        log("[A.verify] no encoded image available; using 16 synthetic raw-id rows")

    with torch.no_grad():
        # (b) new understanding-path feedback (per row)
        new_feat = []
        for r in range(raw.shape[0]):
            new_feat.append(feedback_embed(raw[r:r + 1], bridge, mm_embed_rows))
        new_feat = torch.cat(new_feat, dim=0)  # [N, HIDDEN]

        # (a) old VisualEmbeddingBridge feedback (the parallel reconstruction)
        vt = raw.reshape(raw.shape[0], 1, NUM_CB)         # [N,1,8]
        old_feat = bridge(vt).reshape(raw.shape[0], HIDDEN).to(torch.bfloat16)

    diff = (new_feat.float() - old_feat.float())
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel = (diff.abs().sum() / old_feat.float().abs().sum().clamp(min=1e-9)).item()
    log(f"[A.verify] OLD-bridge vs NEW-understanding feedback: "
        f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rel_L1={rel:.6e}")
    log(f"[A.verify]   new mean|x|={new_feat.float().abs().mean():.4f} "
        f"old mean|x|={old_feat.float().abs().mean():.4f}")
    identical = max_abs < 1e-2
    log(f"[A.verify]   => {'IDENTICAL (old feedback was NOT the bug)' if identical else 'DIFFER (old feedback WAS a bug)'}")
    return identical


# ============================================================================
# 2. Depth-head sampling (mirrors output_processor.depth_transformer_forward_new IMAGE branch)
# ============================================================================
def sample_logits(logits, temp, top_p, top_k):
    # logits: [bs, V]
    if temp <= 0:
        return logits.argmax(dim=-1)   # greedy
    logits = logits.float() / temp
    if top_k is not None and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    # top-p
    sp, si = torch.sort(probs, descending=True, dim=-1)
    csum = torch.cumsum(sp, dim=-1)
    mask = csum - sp > top_p
    sp = sp.masked_fill(mask, 0.0)
    sp = sp / sp.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(sp, 1)
    return si.gather(-1, idx).squeeze(-1)


@torch.no_grad()
def gen_codebooks(hidden, bridge, head, codebook_sizes):
    # hidden: [1, HIDDEN]
    bs = hidden.shape[0]
    next_ids = torch.zeros(bs, NUM_CB, dtype=torch.long, device="cuda")
    emb_layers = bridge.embedding_layers
    for i in range(NUM_CB):
        logits = head(hidden, next_ids, emb_layers, bs, i)     # [bs, CB+1]
        logits[:, codebook_sizes[i]] = torch.finfo(logits.dtype).min  # mask EOS slot 16384
        next_ids[:, i] = sample_logits(logits, TEMP, TOP_P, TOP_K_LEVELS[i] if TOP_K_LEVELS else TOP_K)
    return next_ids   # [1,8] raw ids 0..16383


@torch.no_grad()
def gen_codebooks_cfg(hidden_cond, hidden_uncond, bridge, head, codebook_sizes, cfg_scale):
    """FIX 2 — CFG depth-head. Batches cond+uncond as bs=2 into the image head so each
    codebook level's depth-head input embedding reflects the shared (guided) prior ids.
    Mirrors output_processor.depth_transformer_forward_new IMAGE branch:
      guided = cfg*(cond-uncond)+uncond ; sample from guided ; force SAME id into both.
    Returns [1, NUM_CB] raw ids (cond row == uncond row by construction)."""
    hidden = torch.cat([hidden_cond, hidden_uncond], dim=0)   # [2, HIDDEN]; row0=cond,row1=uncond
    bs = 2
    next_ids = torch.zeros(bs, NUM_CB, dtype=torch.long, device="cuda")
    emb_layers = bridge.embedding_layers
    for i in range(NUM_CB):
        logits = head(hidden, next_ids, emb_layers, bs, i)     # [2, CB+1]
        cond_logits = logits[0:1]
        uncond_logits = logits[1:2]
        guided = cfg_scale * (cond_logits - uncond_logits) + uncond_logits
        guided[:, codebook_sizes[i]] = torch.finfo(guided.dtype).min  # mask EOS slot 16384
        tok = sample_logits(guided, TEMP, TOP_P, TOP_K_LEVELS[i] if TOP_K_LEVELS else TOP_K)        # [1]
        # force SAME sampled token into BOTH cond and uncond (else later codebooks diverge)
        next_ids[0, i] = tok
        next_ids[1, i] = tok
    return next_ids[0:1]   # [1,8]


# ============================================================================
# 3. MAIN — build ModelRunner, prefill, drive the outer image loop
# ============================================================================
def main():
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs, PortArgs
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode
    from sglang.srt.managers.schedule_batch import ScheduleBatch, Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer
    from types import SimpleNamespace

    override = {
        "architectures": ["LongcatFlashForCausalLM"],
        "use_ngram_embedding": True,
        "ngram_embedding_m": 10223616, "ngram_embedding_n": 5, "ngram_embedding_k": 3,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
        "disable_quant_module": ["self_attn"],
    }
    server_args = ServerArgs(
        model_path=NVFP4,
        tokenizer_path=NVFP4,
        quantization="modelopt_fp4",
        json_model_override_args=json.dumps(override),
        attention_backend="flashinfer",
        mem_fraction_static=0.5,      # ~60GB, safely under ceiling (HARD RULE 4)
        max_total_tokens=4096,
        dtype="bfloat16",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        skip_server_warmup=True,
        trust_remote_code=True,
        context_length=4096,
        chunked_prefill_size=4096,    # ngram embedding requires chunked prefill enabled
        device="cuda",
    )
    log("[A.runner] building ModelConfig ...")
    model_config = ModelConfig.from_server_args(server_args)
    port_args = PortArgs.init_new(server_args)
    log("[A.runner] building ModelRunner (mem_fraction_static=0.5) ...")
    mr = ModelRunner(
        model_config=model_config, mem_fraction_static=0.5,
        gpu_id=0, tp_rank=0, tp_size=1, moe_ep_rank=0, moe_ep_size=1,
        pp_rank=0, pp_size=1, nccl_port=port_args.nccl_port, server_args=server_args,
    )
    log(f"[A.runner] ModelRunner up. max_total_num_tokens={mr.max_total_num_tokens}")
    tok = get_tokenizer(NVFP4, trust_remote_code=True)

    # ---- build generation modules (bridge + head + mm_embed_rows) ----
    bridge, head, codebook_sizes, mm_embed_rows = build_gen_modules()
    log("=== MILESTONE 1a: ModelRunner + gen modules loaded ===")

    # ---- FIX 1 verification: old VisualEmbeddingBridge feedback vs understanding path ----
    feedback_identical = verify_feedback_match(bridge, mm_embed_rows)
    log("=== MILESTONE 1b: feedback embed-match verification done ===")

    # CFG toggle (env RUN_CFG=0 -> cond-only to isolate FIX 1; default ON)
    run_cfg = os.environ.get("RUN_CFG", "1") != "0" and USE_CFG
    log(f"[A.cfg] run_cfg={run_cfg} cfg_scale={CFG_SCALE}")

    # tracking free mem
    def memfree():
        fa, ta = torch.cuda.mem_get_info()
        return fa / 1e9, ta / 1e9
    fa, ta = memfree(); log(f"[A.mem] cuda free {fa:.1f}/{ta:.1f} GB")

    model = mr.model

    # === KV-cache-managed driver via ScheduleBatch ===
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator  # noqa
    class _TC(SimpleNamespace):
        def supports_swa(self): return False
        def supports_mamba(self): return False
        def is_chunk_cache(self): return False
        def is_tree_cache(self): return True
        def evict(self, *a, **k): pass
    dummy_tc = _TC(page_size=server_args.page_size, device=mr.device,
                   token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator)

    # Build 1 (cond-only) or 2 (cond+uncond CFG) reqs sharing one ScheduleBatch.
    # Each req carries its own KV cache; bs is 1 or 2 throughout.
    if run_cfg:
        prompts = [PROMPT_COND, PROMPT_UNCOND]
    else:
        prompts = [PROMPT_COND]
    reqs = []
    for ri, ptext in enumerate(prompts):
        pids = tok.encode(ptext)
        log(f"[A.prompt] req{ri} ({'cond' if ri == 0 else 'uncond'}) "
            f"{len(pids)} tokens; tail={pids[-6:]}")
        spp = SamplingParams(temperature=0, max_new_tokens=400)
        spp.normalize(tok)
        rq = Req(rid=ri, origin_input_text=ptext, origin_input_ids=list(pids), sampling_params=spp)
        rq.fill_ids = list(pids)
        rq.logprob_start_len = -1
        rq.set_extend_input_len(len(rq.fill_ids) - len(rq.prefix_indices))
        reqs.append(rq)
    BS = len(reqs)

    batch = ScheduleBatch.init_new(
        reqs=reqs, req_to_token_pool=mr.req_to_token_pool,
        token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
        tree_cache=dummy_tc, model_config=mr.model_config,
        enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE,
    )

    # ===== ngram token-table plumbing (replicates scheduler) =====
    from sglang.jit_kernel.ngram_embedding import update_token_table
    NE_N = mr.model_config.hf_config.ngram_embedding_n
    token_table = mr.token_table
    # multimodal special tokens (text_vocab..text_vocab_plus_mm) are zeroed in table
    tv = getattr(mr.model_config.hf_config, "text_vocab_size", 131072)
    tvp = getattr(mr.model_config.hf_config, "text_vocab_plus_multimodal_special_token_size", 131125)
    zero_lo, zero_hi = tv, tvp  # ids in [zero_lo, zero_hi) -> 0 in table

    def _zero_mm(t):
        if zero_hi > zero_lo:
            m = (t >= zero_lo) & (t < zero_hi)
            t = t.clone()
            t[m] = 0
        return t

    def prefill_token_table():
        batch.ne_token_table = token_table
        all_tokens, column_starts, req_lens = [], [], []
        for rq in batch.reqs:
            start = len(rq.prefix_indices)
            end = start + rq.extend_input_len
            fill_ids = rq.origin_input_ids + rq.output_ids
            if start == 0:
                toks = fill_ids[start:end]; cs = 0
            elif start < NE_N:
                toks = fill_ids[0:end]; cs = 0
            else:
                toks = fill_ids[start - NE_N + 1:end]; cs = start - NE_N + 1
            all_tokens.extend(toks); column_starts.append(cs); req_lens.append(len(toks))
        dev = token_table.device
        tt = _zero_mm(torch.tensor(all_tokens, dtype=token_table.dtype, device=dev))
        update_token_table(
            ne_token_table=token_table, tokens=tt,
            row_indices=batch.req_pool_indices,
            column_starts=torch.tensor(column_starts, dtype=torch.int32, device=dev),
            req_lens=torch.tensor(req_lens, dtype=torch.int32, device=dev),
            ignore_tokens=None,
        )

    def append_token_table(fb, next_tok_id):
        # append the placeholder token at column seq_lens (mm specials -> 0), one per req
        dev = token_table.device
        nrows = fb.req_pool_indices.shape[0]
        tt = _zero_mm(torch.full((nrows,), next_tok_id, dtype=torch.int32, device=dev))
        update_token_table(
            ne_token_table=token_table, tokens=tt,
            row_indices=fb.req_pool_indices,
            column_starts=fb.seq_lens.to(torch.int32),
            req_lens=torch.ones_like(fb.seq_lens, dtype=torch.int32),
            ignore_tokens=None,
        )

    @torch.no_grad()
    def run_forward(input_embeds_override=None):
        mwb = batch.get_model_worker_batch()
        mwb.ne_token_table = token_table
        fb = ForwardBatch.init_new(mwb, mr)
        fb.capture_hidden_mode = CaptureHiddenMode.LAST
        # initialize attention backend metadata (mr.forward() normally does this)
        mr.attn_backend.init_forward_metadata(fb)
        # call the model directly so we can inject input_embeds (A1 feedback)
        if input_embeds_override is not None:
            logits_output = model.forward(None, fb.positions, fb, input_embeds=input_embeds_override)
        else:
            logits_output = model.forward(fb.input_ids, fb.positions, fb, input_embeds=None)
        return logits_output, fb

    def split_hidden(lo):
        """lo.hidden_states [BS, HIDDEN] LAST -> (hidden_cond[1,H], hidden_uncond[1,H] or None)."""
        h = lo.hidden_states.reshape(BS, HIDDEN).to(torch.bfloat16)
        hc = h[0:1]
        hu = h[1:2] if BS == 2 else None
        return hc, hu

    def depth_step(hc, hu):
        """Produce [1,NUM_CB] raw ids, with CFG mixing if BS==2."""
        if BS == 2:
            return gen_codebooks_cfg(hc, hu, bridge, head, codebook_sizes, CFG_SCALE)
        return gen_codebooks(hc, bridge, head, codebook_sizes)

    # ---- PREFILL ----
    batch.prepare_for_extend()
    prefill_token_table()
    lo, fb = run_forward()
    log(f"[A.prefill] hidden {tuple(lo.hidden_states.shape)} dtype {lo.hidden_states.dtype} BS={BS}")
    log("=== MILESTONE 1: prefill returned a hidden ===")
    hc, hu = split_hidden(lo)

    # first depth head (consumes <longcat_img_start> tail hidden -> first image position)
    cur_ids = depth_step(hc, hu)
    log(f"[A.depth] first 8 ids: {cur_ids.tolist()[0]}")
    log("=== MILESTONE 2: depth head -> 8 ids on first hidden ===")

    # ---- OUTER IMAGE LOOP ----
    # Layout: 18 rows; each row = 18 image positions then 1 newline slot.
    all_ids = []  # list of [8] tensors, len 324
    fa, ta = memfree(); log(f"[A.mem] before loop cuda free {fa:.1f} GB")

    def feed_for(ids):
        # FIX 1: feedback via proven understanding mechanism (mm_embed_rows + pre_buffer).
        # Same raw ids fed into BOTH cond and uncond sequences (each advances own KV).
        with torch.no_grad():
            fe = feedback_embed(ids, bridge, mm_embed_rows)  # [1, HIDDEN]
        return fe.expand(BS, HIDDEN).contiguous()

    _prev_h = None
    pos_count = 0
    for row in range(TOKENS_H):
        for col in range(TOKENS_W):
            all_ids.append(cur_ids.reshape(NUM_CB).clone())
            pos_count += 1
            feed_emb = feed_for(cur_ids)                      # [BS, HIDDEN]
            # advance one DECODE step feeding img_pad placeholder but overriding embed.
            batch.output_ids = torch.full((BS,), IMG_PAD, dtype=torch.long, device=mr.device)
            batch.prepare_for_decode()
            lo, fbi = run_forward(input_embeds_override=feed_emb)
            append_token_table(fbi, IMG_PAD)   # keep table columns advanced (->0) for all reqs
            hc, hu = split_hidden(lo)
            if pos_count < TOKENS_H * TOKENS_W:
                cur_ids = depth_step(hc, hu)
            if os.environ.get("DIAG") and pos_count <= int(os.environ.get("DIAG_N","40")):
                import torch.nn.functional as _F
                _h = hc.flatten().float()
                _cs = _F.cosine_similarity(_h, _prev_h, dim=0).item() if _prev_h is not None else float("nan")
                log(f"[DIAG] pos={pos_count} positions={fbi.positions.tolist()} feedAbs={feed_emb.float().abs().mean():.4f} hidAbs={_h.abs().mean():.4f} cosPrev={_cs:.4f} ids={cur_ids.reshape(-1).tolist()}")
                _prev_h = _h.clone()
            if pos_count % 50 == 0:
                fa, _ = memfree()
                log(f"[A.loop] pos {pos_count}/324  cuda free {fa:.1f} GB")
                if fa < 1.5:
                    log("[A.ABORT] cuda free < 1.5GB during loop"); raise RuntimeError("mem abort")
        # newline slot: feed IMG_NEWLINE token via the NORMAL (ngram) embedding path.
        batch.output_ids = torch.full((BS,), IMG_NEWLINE, dtype=torch.long, device=mr.device)
        batch.prepare_for_decode()
        lo, fbi = run_forward(input_embeds_override=None)
        append_token_table(fbi, IMG_NEWLINE)   # ->0 in table for all reqs
        hc, hu = split_hidden(lo)
        # after newline, the next position is an image position -> generate its codebooks
        if pos_count < TOKENS_H * TOKENS_W:
            cur_ids = depth_step(hc, hu)

    gen_ids = torch.stack(all_ids, dim=0).cpu()  # [324,8]
    log(f"[A.done] gen_ids {tuple(gen_ids.shape)}")
    log("=== MILESTONE 3: full loop -> [324,8] ===")
    # stats
    log(f"[A.stats] min={int(gen_ids.min())} max={int(gen_ids.max())}")
    for lvl in range(NUM_CB):
        u = torch.unique(gen_ids[:, lvl])
        log(f"  level {lvl}: unique={u.numel()} range=[{int(u.min())},{int(u.max())}]")
    _tag = os.environ.get("GEN_TAG", "")
    _out = f"/tmp/gen_ids_{_tag}.pt" if _tag else "/tmp/gen_ids.pt"
    torch.save(gen_ids, _out)
    log(f"=== MILESTONE 4: saved {_out} ===")
    fa, ta = memfree(); log(f"[A.mem] end cuda free {fa:.1f}/{ta:.1f} GB")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
