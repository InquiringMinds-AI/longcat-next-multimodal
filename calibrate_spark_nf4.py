#!/usr/bin/env python3
"""
Spark-local NVFP4 input-scale calibration for LongCat-Next (multimodal).

Loads the local NF4 (bitsandbytes 4-bit) checkpoint, registers forward-pre
hooks on backbone leaf modules (skipping the BF16 multimodal towers/heads),
runs the strong real-data calibration sequences through the full multimodal
forward (MockStatus monkey-patch), records max|input| per module, and writes
input_scales.json using the proven formula:  max_abs / (6.0 * 448.0).

Reuses hook + formula + JSON-write logic from calibrate_input_scales.py and
the MockStatus patched_forward from cloud_calibrate_longcat.py.

MEMORY SAFETY: aborts cleanly if `free` available RAM drops below ABORT_GIB.
"""
import json
import os
import sys
import time
import gc
from collections import defaultdict
from dataclasses import dataclass

import torch

NF4_PATH      = os.path.expanduser("~/models/output/LongCat-Next-NF4/")
CALIB_PATH    = os.path.expanduser("~/scripts/longcat-calibration/calib_real_sequences.pt")
EXISTING_JSON = os.path.expanduser("~/models/output/LongCat-Next-NVFP4/input_scales.json")
OUT_DIR       = os.path.expanduser("~/models/output/LongCat-Next-NVFP4-multicalib/")
OUT_JSON      = os.path.join(OUT_DIR, "input_scales.json")

MAXBOUND   = 6.0      # FP4 E2M1 max
SCALE_DEN  = MAXBOUND * 448.0
ABORT_GIB  = 12.0     # abort if available RAM drops below this
MAX_TOKENS = 2048     # truncate very long sequences to bound peak memory

SKIP_PATTERNS = ["visual_", "audio_", "image_", "speech_", "visual_head", "audio_head"]


def avail_gib():
    """Available memory in GiB from /proc/meminfo (MemAvailable)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    return float("inf")


def log_mem(tag):
    print(f"    [mem] {tag}: MemAvailable={avail_gib():.1f} GiB", flush=True)


def check_abort(tag):
    a = avail_gib()
    if a < ABORT_GIB:
        print(f"!!! ABORT at {tag}: MemAvailable {a:.1f} GiB < {ABORT_GIB} GiB floor", flush=True)
        raise MemoryError(f"memory floor breached at {tag}: {a:.1f} GiB")
    return a


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    print("=" * 64)
    print("LongCat-Next NF4 -> NVFP4 input-scale calibration (Spark-local)")
    print("=" * 64)
    log_mem("startup")
    print(f"    torch {torch.__version__}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n[1/5] Loading tokenizer + NF4 model from {NF4_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(NF4_PATH, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        NF4_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    load_s = time.time() - t0
    print(f"    Model loaded in {load_s:.0f}s", flush=True)
    log_mem("after load")
    check_abort("after load")

    # --- MockStatus multimodal monkey-patch (from cloud_calibrate_longcat.py) ---
    @dataclass
    class MockStatus:
        mode: str = "text"
        current_image_token_num: int = -1
        is_audio_text_end: bool = False
        is_audio_start: bool = False
        last_step_mode: str = None
        is_img_newline: bool = False
        is_img_end: bool = False

    mock_status = MockStatus()
    orig_forward = model.forward

    def patched_forward(*args, **kwargs):
        if "multimodal_generation_status" not in kwargs:
            kwargs["multimodal_generation_status"] = mock_status
        return orig_forward(*args, **kwargs)

    model.forward = patched_forward
    model.__call__ = patched_forward

    # --- Register hooks (reused logic from calibrate_input_scales.py) ---
    print("\n[2/5] Registering forward-pre hooks on backbone leaf modules", flush=True)
    max_abs_inputs = defaultdict(float)
    activation_counts = defaultdict(int)
    hooks = []

    try:
        import bitsandbytes as bnb
        Linear4bit = bnb.nn.Linear4bit
    except Exception:
        Linear4bit = None

    for name, module in model.named_modules():
        if any(p in name for p in SKIP_PATTERNS):
            continue
        has_weight = hasattr(module, "weight") and isinstance(
            getattr(module, "weight", None), (torch.Tensor, torch.nn.Parameter)
        )
        if not has_weight and Linear4bit is not None and isinstance(module, Linear4bit):
            has_weight = True
        if not has_weight:
            continue
        if len(list(module.children())) > 0:  # leaf modules only
            continue

        def make_hook(layer_name):
            def hook_fn(mod, inputs):
                if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                    val = inputs[0].detach().float().abs().max().item()
                    if val > max_abs_inputs[layer_name]:
                        max_abs_inputs[layer_name] = val
                    activation_counts[layer_name] += 1
            return hook_fn

        hooks.append(module.register_forward_pre_hook(make_hook(name)))

    n_hooks = len(hooks)
    print(f"    Registered hooks on {n_hooks} leaf modules", flush=True)

    # --- Load calibration sequences ---
    print(f"\n[3/5] Loading calibration sequences from {CALIB_PATH}", flush=True)
    seqs = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    print(f"    {len(seqs)} sequences", flush=True)
    device = next(model.parameters()).device
    print(f"    model device: {device}", flush=True)

    # --- Run calibration ---
    print(f"\n[4/5] Running forward passes (batch=1, no_grad)", flush=True)
    min_avail = avail_gib()
    processed = 0
    errors = 0
    t0 = time.time()
    with torch.no_grad():
        for i, inp in enumerate(seqs):
            if not isinstance(inp, torch.Tensor):
                continue
            if inp.dim() == 1:
                inp = inp.unsqueeze(0)
            if inp.shape[-1] > MAX_TOKENS:
                inp = inp[..., :MAX_TOKENS]
            try:
                model(inp.to(device))
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"    seq {i} (len {inp.shape[-1]}) error: "
                          f"{type(e).__name__}: {e}", flush=True)
            processed += 1

            a = avail_gib()
            if a < min_avail:
                min_avail = a
            if a < ABORT_GIB:
                print(f"!!! ABORT mid-run at seq {i}: MemAvailable {a:.1f} GiB", flush=True)
                for h in hooks:
                    h.remove()
                raise MemoryError(f"memory floor breached mid-run: {a:.1f} GiB")

            if processed % 50 == 0:
                print(f"    processed {processed}/{len(seqs)}  "
                      f"MemAvailable={a:.1f} GiB  min={min_avail:.1f} GiB", flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    run_s = time.time() - t0
    print(f"    Calibration forward complete: {processed} processed, "
          f"{errors} errored, {run_s:.0f}s", flush=True)
    print(f"    min MemAvailable during run: {min_avail:.1f} GiB", flush=True)

    for h in hooks:
        h.remove()

    n_with_data = len(activation_counts)
    print(f"    modules that recorded >=1 activation: {n_with_data}/{n_hooks}", flush=True)

    # --- Compute scales (reused formula) ---
    print("\n[5/5] Computing input_scales and writing JSON", flush=True)
    input_scales = {}
    cold = 0
    for name, max_abs in sorted(max_abs_inputs.items()):
        if max_abs > 0:
            input_scales[name] = max_abs / SCALE_DEN
        else:
            input_scales[name] = 1.0 / SCALE_DEN
            cold += 1

    # Modules hooked but never activated (count == 0) get the cold default too,
    # so the key set is complete relative to hooked modules.
    hooked_names = set()
    for name, module in model.named_modules():
        if any(p in name for p in SKIP_PATTERNS):
            continue
        has_weight = hasattr(module, "weight") and isinstance(
            getattr(module, "weight", None), (torch.Tensor, torch.nn.Parameter))
        if not has_weight and Linear4bit is not None and isinstance(module, Linear4bit):
            has_weight = True
        if not has_weight or len(list(module.children())) > 0:
            continue
        hooked_names.add(name)

    no_activation = sorted(hooked_names - set(activation_counts.keys()))
    for name in no_activation:
        if name not in input_scales:
            input_scales[name] = 1.0 / SCALE_DEN

    with open(OUT_JSON, "w") as f:
        json.dump(input_scales, f, indent=2)
    print(f"    Wrote {len(input_scales)} keys -> {OUT_JSON}", flush=True)

    # --- Validate against existing ---
    print("\n=== VALIDATION ===", flush=True)
    vals = list(input_scales.values())
    finite = all((v == v) and v != float("inf") and v != float("-inf") for v in vals)
    positive = all(v > 0 for v in vals)
    print(f"    keys written        : {len(input_scales)}")
    print(f"    all finite          : {finite}")
    print(f"    all positive        : {positive}")
    print(f"    value min/max/mean  : {min(vals):.6e} / {max(vals):.6e} / "
          f"{sum(vals)/len(vals):.6e}")
    print(f"    modules hooked      : {n_hooks}")
    print(f"    modules w/ data     : {n_with_data}")
    print(f"    cold (max_abs==0)   : {cold}")
    print(f"    hooked-but-no-activ : {len(no_activation)}")
    print(f"    sequences processed : {processed} ({errors} errored)")
    print(f"    peak (min avail) RAM: {min_avail:.1f} GiB")

    if os.path.exists(EXISTING_JSON):
        existing = json.load(open(EXISTING_JSON))
        ek = set(existing.keys())
        nk = set(input_scales.keys())
        print(f"\n    existing keys       : {len(ek)}")
        print(f"    new keys            : {len(nk)}")
        print(f"    shared keys         : {len(ek & nk)}")
        print(f"    only-in-new         : {len(nk - ek)}")
        print(f"    only-in-existing    : {len(ek - nk)}")
        sample = ["model.embed_tokens",
                  "model.layers.0.self_attn.0.q_b_proj",
                  "model.layers.0.self_attn.0.kv_b_proj"]
        print("    sample comparison (key: new vs existing):")
        for k in sample:
            n = input_scales.get(k)
            e = existing.get(k)
            print(f"      {k}: {n} vs {e}")

    if no_activation:
        print(f"\n    !! {len(no_activation)} hooked modules got NO activations:")
        for nm in no_activation[:40]:
            print(f"       {nm}")
        if len(no_activation) > 40:
            print(f"       ... (+{len(no_activation) - 40} more)")

    print(f"\nTotal wall time: {time.time() - t_start:.0f}s", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
