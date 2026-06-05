#!/usr/bin/env python3
"""Push the surviving calib sequences through the (SQ_CALIB=1) server so the MoE
forward accumulates per-channel activation maxima → /tmp/sq_actmax.pt."""
import torch, requests, time
seqs = torch.load("/cal/calib_real_sequences.pt", map_location="cpu", weights_only=False)
print(f"[calib] {len(seqs)} sequences", flush=True)
ids = [s.reshape(-1).tolist() for s in seqs]
B = 32
t0 = time.time()
for i in range(0, len(ids), B):
    batch = ids[i:i+B]
    # send as independent requests in one HTTP call (sglang accepts a list of input_ids)
    r = requests.post("http://localhost:8090/generate",
                      json={"input_ids": batch,
                            "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
                      timeout=300)
    print(f"[calib] {min(i+B,len(ids))}/{len(ids)}  HTTP {r.status_code}", flush=True)
print(f"[calib] done in {time.time()-t0:.0f}s", flush=True)
