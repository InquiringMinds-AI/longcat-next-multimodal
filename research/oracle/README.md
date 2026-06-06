# oracle/ — the bnb-int8 capability proof + the probes

These scripts run the **BF16 LongCat-Next loaded as bitsandbytes-int8** via HuggingFace
`generate()` — the path that first proved every modality *can* work at 8-bit on a GB10.
They are **reference and proof tooling, not the serving path** (the server is at the repo
root). Kept because each one backs a specific claim in [../FINDINGS.md](../FINDINGS.md).

| script | what it proves |
|---|---|
| `q8_unified.py` | The capability proof: **one** 8-bit load routes all five task types (text, image/audio understanding, image gen, audio gen). The "precision was the lever" result, end to end. |
| `teacher_force_image.py` | The turn in Act I: feeds a real image's `[324,8]` RVQ codes as history and measures the depth head's argmax vs. truth → the conditional + head are **sound given correct history**; the free-run failure was autoregressive drift, not content-blindness. |
| `decode_roundtrip.py` | Decode-stack elimination (image): real image → encode → decode reconstructs cleanly → decoder / refiner / VQ codebooks are sound. |
| `audio_decode_roundtrip.py` | Decode-stack elimination (audio): the same round-trip for the RVQ audio tokenizer + vocoder. |

> These target the BF16 weights at `~/models/LongCat-Next` inside the cu130 SGLang
> container, with the bnb-int8 dependency pin described in FINDINGS.md (Act I). They are not
> maintained as part of the serving package and are not covered by the root self-test.
