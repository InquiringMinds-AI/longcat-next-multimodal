# research/ — how this was built

The engineering journey behind the server at the repo root. The shipped artifact is the
*what*; this is the *how* and the *why*.

- **[FINDINGS.md](FINDINGS.md)** — the narrative. Two debugging wins that looked like the
  same bug (incoherent generation) but weren't: a 4-bit **precision floor**, then a
  **structural** omission (a dropped spatial anchor) that only surfaced at 8-bit on a real
  server — plus the adversarial review that caught a silent MoE scaling bug. Start here.

- **[oracle/](oracle/)** — the bitsandbytes-int8 capability proof and the probes that
  backed the narrative's key claims (teacher-forcing soundness, decode-stack elimination).
  These run the BF16 model via HuggingFace `generate()`, not the SGLang server — they're
  reference/proof tooling, not the serving path.

The earlier **4-bit NVFP4 SGLang port** and the **streaming calibration** tooling are
preserved in git history at commit `be21cc8` (the pre-restructure snapshot) — the ancestors
of the shipped `new_files/` overlay. See the closing section of FINDINGS.md.
