#!/usr/bin/env python3
"""
Streaming per-task NVFP4 input-scale calibration for LongCat-Next on Spark.

Why this exists: HF from_pretrained loads the full state-dict into (unified)
RAM and copies to GPU simultaneously -> ~120GB peak on GB10's 128GB shared
pool -> crash-to-OFF. This loader builds the skeleton on `meta`, then streams
each weight DIRECTLY to GPU from safetensors (zero-copy mmap slices), freeing
as it goes. Peak ~= model size, no CPU doubling.

Towers (visual/audio tokenizers + heads) are stubbed nn.Identity in the
modeling code (provably unused: codebook tokens embed via embed_tokens[282624];
verified). So only the backbone loads.

DRY_RUN=1 -> validate key mapping (metadata only, NO GPU, NO weight load).
"""
import os, json, time, gc, math
from collections import defaultdict
from dataclasses import dataclass
import torch
import torch.nn as nn

NF4_ST   = os.path.expanduser("~/models/output/LongCat-Next-NF4-st/")
CALIB    = os.path.expanduser(os.environ.get("CALIB_PATH", "~/scripts/longcat-calibration/calib_real_sequences.pt"))
OUT_DIR  = os.path.expanduser("~/models/output/per_task_calib/")
SCALE_DEN = 6.0 * 448.0
MAX_TOKENS = 2048
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
SKIP_PATTERNS = ["visual_tokenizer", "audio_tokenizer", "visual_head", "audio_head", "image_", "speech_"]
TASKS = {"textgen": (0, 60), "imggen": (60, 280), "audgen": (500, 554)}
QSUF = ["absmax", "nested_absmax", "nested_quant_map", "quant_map", "quant_state.bitsandbytes__nf4"]

def avail_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0*1024.0)
    return float("inf")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("="*64); print(f"STREAMING per-task calibration  (DRY_RUN={DRY_RUN})"); print("="*64, flush=True)

    idx = json.load(open(NF4_ST + "model.safetensors.index.json"))
    wm = idx["weight_map"]
    is_4bit_weight = lambda k: (k + ".quant_state.bitsandbytes__nf4") in wm
    quant_data_keys = {k for k in wm if is_4bit_weight(k)}
    aux_suffixes = tuple("." + s for s in QSUF)
    aux_keys = {k for k in wm if k.endswith(aux_suffixes)}
    plain_keys = {k for k in wm if k not in quant_data_keys and k not in aux_keys}
    print(f"[idx] total={len(wm)} 4bit-weights={len(quant_data_keys)} aux={len(aux_keys)} plain={len(plain_keys)}", flush=True)

    from transformers import AutoConfig, AutoModelForCausalLM
    # transformers 5.x renamed Qwen2RMSNorm; remote tower code imports the old name
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q25
    if not hasattr(_q25, 'Qwen2RMSNorm') and hasattr(_q25, 'Qwen2_5_VLRMSNorm'):
        _q25.Qwen2RMSNorm = _q25.Qwen2_5_VLRMSNorm
    config = AutoConfig.from_pretrained(NF4_ST, trust_remote_code=True)
    print("[build] constructing skeleton on meta ...", flush=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()

    # Identify which leaf nn.Linear modules are 4-bit per the checkpoint.
    lin_names = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    to_4bit = {n for n in lin_names if (n + ".weight") in quant_data_keys}
    plain_lin = {n for n in lin_names if (n + ".weight") in plain_keys}
    backbone_4bit = {n for n in to_4bit if not any(p in n for p in SKIP_PATTERNS)}
    skipped_4bit = to_4bit - backbone_4bit
    print(f"[map] nn.Linear={len(lin_names)} -> 4bit={len(to_4bit)} (backbone {len(backbone_4bit)}, skipped {len(skipped_4bit)}) plain_lin={len(plain_lin)}", flush=True)

    # Which checkpoint keys will we actually load (backbone only)?
    def is_backbone(k):
        return not any(p in k for p in SKIP_PATTERNS)
    load_4bit = {k for k in quant_data_keys if is_backbone(k)}
    load_plain = {k for k in plain_keys if is_backbone(k)}
    skip_keys = (quant_data_keys | plain_keys) - load_4bit - load_plain
    print(f"[plan] load 4bit-weights={len(load_4bit)} plain={len(load_plain)}  skip(tower)={len(skip_keys)}", flush=True)

    # meta params that must end up materialized (backbone), minus stubbed towers
    meta_params = [(n, p) for n, p in model.named_parameters()]
    meta_bufs = [(n, b) for n, b in model.named_buffers()]
    backbone_param_names = {n for n, _ in meta_params if is_backbone(n)}
    print(f"[skeleton] params={len(meta_params)} buffers={len(meta_bufs)} backbone-params={len(backbone_param_names)}", flush=True)

    # Cross-check: every backbone param has a source key (as 4bit data or plain)
    unmatched = []
    for n, p in meta_params:
        if not is_backbone(n):
            continue
        if n in load_plain:
            continue
        # 4bit weight: module.weight where module in backbone_4bit
        if n.endswith(".weight") and n[:-len(".weight")] in backbone_4bit:
            continue
        unmatched.append(n)
    print(f"[check] backbone params WITHOUT a source key: {len(unmatched)}", flush=True)
    for n in unmatched[:20]:
        print("    UNMATCHED:", n, flush=True)

    if DRY_RUN:
        print("\n[DRY_RUN] mapping validated; no GPU, no weights loaded. Exiting.", flush=True)
        return

    # ---- REAL LOAD (GPU) ----
    import bitsandbytes as bnb
    handles = {}
    def get(key):
        fn = wm[key]
        if fn not in handles:
            handles[fn] = __import__("safetensors").safe_open(NF4_ST + fn, framework="pt", device="cuda")
        return handles[fn].get_tensor(key)
    handles_cpu = {}
    def get_cpu(key):
        fn = wm[key]
        if fn not in handles_cpu:
            handles_cpu[fn] = __import__('safetensors').safe_open(NF4_ST + fn, framework='pt', device='cpu')
        return handles_cpu[fn].get_tensor(key)

    def set_submodule_param(root, dotted, value, is_buffer=False):
        *parents, leaf = dotted.split(".")
        mod = root
        for p in parents:
            mod = getattr(mod, p)
        if is_buffer:
            mod._buffers[leaf] = value
        else:
            mod._parameters[leaf] = value if isinstance(value, nn.Parameter) else nn.Parameter(value, requires_grad=False)

    def get_module(root, dotted):
        mod = root
        for p in dotted.split("."):
            mod = getattr(mod, p)
        return mod

    print("\n[load] streaming backbone weights to GPU ...", flush=True)
    t0 = time.time(); n4 = 0
    # 4-bit modules: build Linear4bit on the fly + from_prequantized
    for name in sorted(backbone_4bit):
        wk = name + ".weight"
        lin = get_module(model, name)
        new_lin = bnb.nn.Linear4bit(lin.in_features, lin.out_features,
                                    bias=lin.bias is not None, compute_dtype=torch.bfloat16,
                                    quant_type="nf4", device="meta")
        data = get(wk)
        qs = {s: get(f"{name}.weight.{s}") for s in QSUF if (f"{name}.weight.{s}") in wm}
        new_lin.weight = bnb.nn.Params4bit.from_prequantized(data=data, quantized_stats=qs, device="cuda", module=new_lin)
        if lin.bias is not None and (name + ".bias") in wm:
            new_lin.bias = nn.Parameter(get(name + ".bias"), requires_grad=False)
        # swap into tree
        *par, leaf = name.split("."); m = model
        for p in par: m = getattr(m, p)
        setattr(m, leaf, new_lin)
        n4 += 1
        if n4 % 1000 == 0:
            print(f"    4bit {n4}/{len(backbone_4bit)}  avail={avail_gib():.1f}GiB", flush=True)
            gc.collect()
    print(f"[load] 4bit done: {n4} modules, {time.time()-t0:.0f}s, avail={avail_gib():.1f}GiB", flush=True)

    # plain params/buffers (backbone): load BF16/int8 to GPU
    npl = 0; nq8 = 0
    name_to_buf = dict(model.named_buffers())
    NGRAM = 'model.ngram_embeddings.embedders.'
    for n in sorted(load_plain):
        if NGRAM in n and n.endswith('.weight'):
            w = get_cpu(n)
            scale = (w.abs().max().float() / 127.0).clamp_min(1e-8)
            w8 = (w.float() / scale).round().clamp_(-127, 127).to(torch.int8).to('cuda')
            mod_name = n[:-len('.weight')]
            *par, leaf = mod_name.split('.'); m = model
            for pp in par: m = getattr(m, pp)
            emb = getattr(m, leaf)
            emb.weight = nn.Parameter(w8, requires_grad=False)
            emb._int8_scale = scale.to('cuda')
            del w; nq8 += 1
            if nq8 % 4 == 0: gc.collect()
        else:
            is_buf = n in name_to_buf
            set_submodule_param(model, n, get(n).to('cuda'), is_buffer=is_buf)
            npl += 1
    print(f'[load] plain done: {npl} tensors, ngram-int8={nq8}, avail={avail_gib():.1f}GiB', flush=True)
    # int8-aware ngram embedder forward (dequant on lookup)
    import torch.nn.functional as _F
    mm0 = model.model if hasattr(model, 'model') else model
    EWM = type(mm0.ngram_embeddings.embedders[0])
    def _ewm_fwd(self, input, mask=None):
        mask = mask.bool() if mask is not None else torch.ones_like(input, dtype=torch.bool)
        bs, sl = input.shape; dim = self.embedding_dim
        scale = getattr(self, '_int8_scale', None)
        if scale is None:
            out = torch.zeros((bs, sl, dim), device=input.device, dtype=self.weight.dtype)
            vi = input[mask]
            if vi.numel() > 0:
                out[mask] = _F.embedding(vi, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
            return out
        out = torch.zeros((bs, sl, dim), device=input.device, dtype=torch.bfloat16)
        vi = input[mask]
        if vi.numel() > 0:
            out[mask] = self.weight[vi].to(torch.bfloat16) * scale
        return out
    EWM.forward = _ewm_fwd
    print('[patch] ngram int8 dequant forward installed', flush=True)

    # recompute multimodal offset buffers on GPU (built on meta, persistent=False)
    mm = model.model if hasattr(model, "model") else model
    if hasattr(mm, "_init_multimodal_constants"):
        mm._init_multimodal_constants(config)
        for bn in ["visual_offset_vals", "audio_offset_vals"]:
            if hasattr(mm, bn) and getattr(mm, bn).device.type != "cuda":
                setattr(mm, bn, getattr(mm, bn).to("cuda"))

    # derived RoPE inv_freq buffers/params (not in checkpoint) -> materialize off meta
    def _nav_parent(name):
        *par, leaf = name.split('.'); mod = model
        for pp in par: mod = getattr(mod, pp)
        return mod, leaf
    def _materialize(name, value):
        mod, leaf = _nav_parent(name)
        if leaf in mod._parameters:
            mod._parameters[leaf] = nn.Parameter(value, requires_grad=False)
        elif leaf in mod._buffers:
            mod._buffers[leaf] = value
        else:
            setattr(mod, leaf, value)
    def _inv_freq_for(name):
        mod, _ = _nav_parent(name)
        if hasattr(mod, 'rope_init_fn'):
            try:
                inv, sc = mod.rope_init_fn(getattr(mod, 'config', config), device='cuda')
                if hasattr(mod, 'attention_scaling'): mod.attention_scaling = sc
                return inv
            except Exception as e:
                print('  rope_init_fn failed:', e, flush=True)
        dim = getattr(config, 'qk_rope_head_dim', getattr(config, 'head_dim', 64))
        base = float(getattr(config, 'rope_theta', 10000.0))
        return 1.0 / (base ** (torch.arange(0, dim, 2, device='cuda').float() / dim))

    metas = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == 'meta']
    for n in metas:
        if n.endswith('inv_freq'):
            _materialize(n, _inv_freq_for(n)); print(f'[fix] materialized {n}', flush=True)

    # plain-attribute (non-registered) tensors built under meta -> materialize
    import torch as _t
    for mod_name, mod in model.named_modules():
        for attr, val in list(vars(mod).items()):
            if isinstance(val, _t.Tensor) and val.device.type == 'meta':
                if attr == 'oe_ignored_token_ids':
                    setattr(mod, attr, _t.tensor(config.oe_ignored_token_ids, dtype=_t.long, device='cuda'))
                    print(f'[fix] materialized attr {mod_name}.{attr}', flush=True)
                else:
                    print(f'[warn] unhandled meta attr {mod_name}.{attr} shape={tuple(val.shape)} dtype={val.dtype}', flush=True)
    # any backbone meta leftovers?
    leftover = [n for n, p in model.named_parameters() if p.device.type == "meta" and not any(s in n for s in SKIP_PATTERNS)]
    leftover += [n for n, b in model.named_buffers() if b.device.type == "meta" and not any(s in n for s in SKIP_PATTERNS)]
    print(f"[verify] backbone meta-leftover tensors: {len(leftover)}", flush=True)
    _allmeta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == 'meta']
    print(f'[diag] ALL meta tensors (incl stubbed): {len(_allmeta)}', flush=True)
    for _n in _allmeta[:40]: print('    META:', _n, flush=True)
    for n in leftover[:15]: print("    LEFTOVER:", n, flush=True)
    if leftover:
        raise RuntimeError(f"{len(leftover)} backbone tensors still on meta - load incomplete")

    # MockStatus for forward
    @dataclass
    class MockStatus:
        mode: str = "text"; current_image_token_num: int = -1
        is_audio_text_end: bool = False; is_audio_start: bool = False
        last_step_mode: str = None; is_img_newline: bool = False; is_img_end: bool = False
    ms = MockStatus(); of = model.forward
    def pf(*a, **k):
        k.setdefault("multimodal_generation_status", ms); return of(*a, **k)
    model.forward = pf; model.__call__ = pf

    # hooks on backbone leaf modules with weights
    max_abs = defaultdict(float); counts = defaultdict(int); hooks = []
    Lin4 = bnb.nn.Linear4bit
    for name, module in model.named_modules():
        if any(p in name for p in SKIP_PATTERNS): continue
        hw = (hasattr(module, "weight") and isinstance(getattr(module, "weight", None), (torch.Tensor, nn.Parameter))) or isinstance(module, Lin4)
        if not hw or len(list(module.children())) > 0: continue
        def mk(nm):
            def h(mod, inp):
                if inp and isinstance(inp[0], torch.Tensor):
                    v = inp[0].detach().float().abs().max().item()
                    if v > max_abs[nm]: max_abs[nm] = v
                    counts[nm] += 1
            return h
        hooks.append(module.register_forward_pre_hook(mk(name)))
    print(f"[hooks] {len(hooks)} backbone leaf modules", flush=True)

    seqs = torch.load(CALIB, map_location="cpu", weights_only=False)
    _stask = os.environ.get("SINGLE_TASK")
    tasks_to_run = {_stask: (0, len(seqs))} if _stask else TASKS
    if _stask: print(f"[single-task] {_stask} over {len(seqs)} seqs", flush=True)
    dev = "cuda"
    results = {}
    for task, (lo, hi) in tasks_to_run.items():
        max_abs.clear(); counts.clear()
        print(f"\n[{task}] seqs[{lo}:{hi}]", flush=True)
        with torch.no_grad():
            for j, inp in enumerate(seqs[lo:hi]):
                if not isinstance(inp, torch.Tensor): continue
                if inp.dim() == 1: inp = inp.unsqueeze(0)
                if inp.shape[-1] > MAX_TOKENS: inp = inp[..., :MAX_TOKENS]
                try: model(inp.to(dev))
                except Exception as e:
                    if j < 1:
                        import traceback; traceback.print_exc()
                    if j < 3: print(f"    seq {j} err {type(e).__name__}: {e}", flush=True)
                if (j+1) % 50 == 0:
                    print(f"    {j+1}/{hi-lo} avail={avail_gib():.1f}GiB", flush=True); gc.collect(); torch.cuda.empty_cache()
        scales = {n: m/SCALE_DEN for n, m in max_abs.items() if m > 0 and counts[n] > 0}
        results[task] = {"scales": scales, "hot": set(scales)}
        json.dump(scales, open(os.path.join(OUT_DIR, f"input_scales_{task}.json"), "w"), indent=2)
        print(f"    {task}: {len(scales)} hot modules", flush=True)
    for h in hooks: h.remove()

    # divergence report
    if not all(k in results for k in ("imggen","audgen","textgen")):
        print(f"[single-task] scales written for {list(results)}; skipping 3-way divergence report.", flush=True)
        print("DONE.", flush=True); return
    print("\n" + "="*64); print("DIVERGENCE REPORT"); print("="*64, flush=True)
    def is_expert(n): return "expert" in n.lower()
    def summarize(label, rd):
        if not rd: print(f"  {label}: none"); return
        vals = sorted(rd.values()); logs = [abs(math.log(r)) for r in vals if r > 0]
        gm = math.exp(sum(math.log(r) for r in vals if r > 0)/len(vals))
        def pct(p): return vals[min(len(vals)-1, int(p*len(vals)))]
        w2 = sum(1 for r in vals if 0.5 <= r <= 2.0)/len(vals)
        b4 = sum(1 for r in vals if r < 0.25 or r > 4.0)/len(vals)
        print(f"  {label}: n={len(vals)} geomean={gm:.3f} |log|mean={sum(logs)/len(logs):.3f} p05={pct(.05):.3f} p50={pct(.5):.3f} p95={pct(.95):.3f} within2x={w2*100:.0f}% beyond4x={b4*100:.0f}%", flush=True)
    for a, b in [("imggen","audgen"),("imggen","textgen"),("audgen","textgen")]:
        sa, sb = results[a]["scales"], results[b]["scales"]
        common = results[a]["hot"] & results[b]["hot"]
        rall = {n: sa[n]/sb[n] for n in common if sb[n] > 0}
        rexp = {n: r for n, r in rall.items() if is_expert(n)}
        print(f"\n--- {a} vs {b}: {len(common)} common hot ---")
        summarize(f"ALL    {a}/{b}", rall)
        summarize(f"EXPERT {a}/{b}", rexp)
    print("\nDONE.", flush=True)

if __name__ == "__main__":
    main()
