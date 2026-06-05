#!/usr/bin/env python3
"""Drive LongCat-Next image generation through the sglang server.
Faithful prefill order (matches HF oracle prepare_inputs_for_generation):
  [prompt tokens] + [anyres_prefix tokens "<longcat_img_token_size>H W</...>"] + [image_start]
The anyres_prefix is the spatial-composition anchor; without it the model
generates locally-plausible texture with no global layout. token grid = 37x37."""
import sys, time, glob, json, requests
from transformers import AutoTokenizer

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "A photograph of an orange tabby cat sitting on a windowsill."
IMG_START = 131106
ANYRES = "<longcat_img_token_size>37 37</longcat_img_token_size>"
tok = AutoTokenizer.from_pretrained("/workspace/model", trust_remote_code=True)
ids = (tok(PROMPT, add_special_tokens=False).input_ids
       + tok(ANYRES, add_special_tokens=False).input_ids
       + [IMG_START])
print("[trigger] prompt=%r total_ids=%d tail=%s (anyres+image_start=%d)" % (PROMPT, len(ids), ids[-9:], IMG_START), flush=True)

before = set(glob.glob("/tmp/longcat_img_*.png"))
t0 = time.time()
r = requests.post("http://localhost:8090/generate",
                  json={"input_ids": ids,
                        "sampling_params": {"max_new_tokens": 1500, "temperature": 0.5, "top_k": 1024, "top_p": 0.75}},
                  timeout=900)
dt = time.time() - t0
print("[trigger] HTTP %d in %ds" % (r.status_code, dt), flush=True)
try:
    j = r.json(); print("[trigger] meta:", json.dumps(j.get("meta_info", {}), ensure_ascii=False)[:300], flush=True)
except Exception as e:
    print("[trigger] resp parse:", e, r.text[:300], flush=True)
new = sorted(set(glob.glob("/tmp/longcat_img_*.png")) - before)
print("[trigger] NEW PNG(s):", new, flush=True)
