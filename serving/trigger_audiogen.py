#!/usr/bin/env python3
"""Drive LongCat-Next voice-clone AUDIO generation through the sglang server.

Canonical spk_syn format (from LongCat-Next-inference/example/test_cases.yaml, with the
YAML '>' folded-scalar SPACING reproduced exactly — spaces before <audio_start>, after
<audio_end>, before <assistant>, plus a trailing newline; unspaced concat makes the model
free-generate instead of reciting SYN_TEXT):

  <longcat_system>Replicate the voice in the audio clip to formulate an answer. \
  <longcat_audio_start><longcat_audio_end> \
  <longcat_user>用这个声音合成以下内容：<SYN_TEXT> \
  <longcat_assistant><longcat_audiogen_start>\n

The empty <longcat_audio_start><longcat_audio_end> span is the in-prompt placeholder; the
(fixed) server processor inserts the reference-voice pad block there from audio_data.
The reference clip is sent via audio_data (a path accessible inside the container)."""
import sys, time, glob, json, requests

SYN_TEXT = sys.argv[1] if len(sys.argv) > 1 else "The quick brown fox jumps over the lazy dog."
REF = "/tmp/spk_syn.wav"

# EXACT canonical format, verified against tok.apply_chat_template() output of the
# bnb q8_audio.py messages: 'answer:' colon, NO inter-segment spaces, NO trailing newline.
# The inline ref path is replaced by an empty <audio_start><audio_end> placeholder; the
# (fixed) processor splices the encoded ref (from audio_data) there — producing token-for-token
# the same input_ids the canonical processor produces from the inline path.
prompt = ("<longcat_system>Replicate the voice in the audio clip to formulate an answer:"
          "<longcat_audio_start><longcat_audio_end>"
          "<longcat_user>用这个声音合成以下内容：" + SYN_TEXT +
          "<longcat_assistant><longcat_audiogen_start>")
# NO trailing whitespace — <longcat_audiogen_start> must be the LAST prefill token or
# _check_prefill_audio_start won't flip into audio mode.
print("[trigger] SYN_TEXT=%r" % SYN_TEXT, flush=True)
print("[trigger] prompt=%r" % prompt, flush=True)

before = set(glob.glob("/tmp/longcat_tts_*.wav"))
t0 = time.time()
r = requests.post("http://localhost:8090/generate",
                  json={"text": prompt,
                        "audio_data": [REF],
                        # NO repetition_penalty: sglang sizes acc_additive_penalties to the
                        # full n-gram OE vocab (282624) but the transcript head emits 131125
                        # logits → logits.add_ size-mismatch crashes the scheduler.
                        "sampling_params": {"max_new_tokens": 1200, "temperature": 0.5,
                                            "top_k": 5, "top_p": 0.85}},
                  timeout=900)
dt = time.time() - t0
print("[trigger] HTTP %d in %ds" % (r.status_code, dt), flush=True)
try:
    j = r.json(); print("[trigger] meta:", json.dumps(j.get("meta_info", {}), ensure_ascii=False)[:300], flush=True)
    print("[trigger] text:", (j.get("text") or "")[:200], flush=True)
except Exception as e:
    print("[trigger] resp parse:", e, r.text[:300], flush=True)
new = sorted(set(glob.glob("/tmp/longcat_tts_*.wav")) - before)
print("[trigger] NEW WAV(s):", new, flush=True)
