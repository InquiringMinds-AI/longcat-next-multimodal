#!/usr/bin/env python3
"""Voice-clone TTS through the running LongCat-Next server.
Run INSIDE the container:  docker exec longcat-next python3 /workspace/scripts/gen_audio.py "text to speak"
Reference voice: REF_WAV env (default /workspace/scripts/voices/en_reference.wav). Writes <LCN_OUTPUT_DIR>/longcat_tts_*.wav.

Canonical spk_syn prompt format (exact, verified against apply_chat_template): 'answer:' colon, NO
inter-segment spaces, empty <audio_start><audio_end> placeholder (the ref voice is supplied via
audio_data and spliced there by the processor), and <audiogen_start> as the LAST token."""
import sys, os, time, glob, json, requests

SYN_TEXT = sys.argv[1] if len(sys.argv) > 1 else "The quick brown fox jumps over the lazy dog."
PORT = os.environ.get("PORT", "8090")
OUT = os.environ.get("LCN_OUTPUT_DIR", "/tmp")
REF = os.environ.get("REF_WAV", "/workspace/scripts/voices/en_reference.wav")
prompt = ("<longcat_system>Replicate the voice in the audio clip to formulate an answer:"
          "<longcat_audio_start><longcat_audio_end>"
          "<longcat_user>用这个声音合成以下内容：" + SYN_TEXT +
          "<longcat_assistant><longcat_audiogen_start>")
print("[gen_audio] SYN_TEXT=%r  ref=%s" % (SYN_TEXT, REF), flush=True)
before = set(glob.glob(f"{OUT}/longcat_tts_*.wav"))
t0 = time.time()
r = requests.post(f"http://localhost:{PORT}/generate",
                  json={"text": prompt, "audio_data": [REF],
                        "sampling_params": {"max_new_tokens": 1200, "temperature": 0.5, "top_k": 5, "top_p": 0.85}},
                  timeout=900)
print("[gen_audio] HTTP %d in %ds" % (r.status_code, time.time() - t0), flush=True)
try:
    print("[gen_audio] transcript:", (r.json().get("text") or "")[:200], flush=True)
except Exception:
    pass
new = sorted(set(glob.glob(f"{OUT}/longcat_tts_*.wav")) - before)
print("[gen_audio] NEW WAV(s):", new, flush=True)
