#!/usr/bin/env python3
"""Text -> image generation through the running LongCat-Next server.
Run INSIDE the container:  docker exec longcat-next python3 /workspace/scripts/gen_image.py "a prompt"
Writes <LCN_OUTPUT_DIR>/longcat_img_*_decoded.png and _refined.png (refined = final)."""
import sys, os, time, glob, requests

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "A photograph of an orange tabby cat sitting on a windowsill."
PORT = os.environ.get("PORT", "8090")
OUT = os.environ.get("LCN_OUTPUT_DIR", "/tmp")
IMG_START = 131106
ANYRES = "<longcat_img_token_size>37 37</longcat_img_token_size>"  # 37x37 grid spatial anchor
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/workspace/model", trust_remote_code=True)
ids = (tok(PROMPT, add_special_tokens=False).input_ids
       + tok(ANYRES, add_special_tokens=False).input_ids
       + [IMG_START])
print("[gen_image] prompt=%r" % PROMPT, flush=True)
before = set(glob.glob(f"{OUT}/longcat_img_*.png"))
t0 = time.time()
r = requests.post(f"http://localhost:{PORT}/generate",
                  json={"input_ids": ids,
                        "sampling_params": {"max_new_tokens": 1500, "temperature": 0.5, "top_k": 1024, "top_p": 0.75}},
                  timeout=900)
print("[gen_image] HTTP %d in %ds" % (r.status_code, time.time() - t0), flush=True)
new = sorted(set(glob.glob(f"{OUT}/longcat_img_*.png")) - before)
print("[gen_image] NEW PNG(s):", new, flush=True)
