#!/usr/bin/env python3
"""Video understanding through the running LongCat-Next server.
Run INSIDE the container:
  docker exec longcat-next python3 /workspace/scripts/understand_video.py /path/in/container.mp4 "your question"
The video is decoded (decord), sampled to frames, and run through the visual encoder.
Mount your video somewhere visible to the container (e.g. into /workspace/outputs)."""
import sys, os, json, requests

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/workspace/outputs/test_video.mp4"
QUESTION = sys.argv[2] if len(sys.argv) > 2 else "Describe what you see in this video."
PORT = os.environ.get("PORT", "8090")
prompt = f"<longcat_user>{QUESTION}<longcat_assistant>"
r = requests.post(f"http://localhost:{PORT}/generate",
                  json={"text": prompt, "video_data": [VIDEO],
                        "sampling_params": {"max_new_tokens": 256, "temperature": 0.3}},
                  timeout=600)
print("HTTP", r.status_code)
try:
    print("DESCRIPTION:", r.json().get("text", ""))
except Exception as e:
    print("err", e, r.text[:300])
