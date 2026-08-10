#!/usr/bin/env python3
"""Do CONCURRENT multimodal requests scatter their embeddings to the wrong positions?

HYPOTHESIS. The processor computes each item's `offsets` relative to ITS OWN request's
input_ids. At forward time `_get_mm_items` flattens the items of every request in the
batch into one list, losing which request each came from, and `_replace_mm_embeddings`
uses those offsets as direct indices into the BATCH-FLATTENED embedding tensor. With one
request in flight the two coincide and everything works. With two multimodal requests in
the same batched prefill, the second request's offsets index into the FIRST request's
span.

This is distinct from the prefix-cache collision fixed earlier: that one served a stale
cached KV for a sequentially-issued request, and its probes were all sequential — so none
of them could have caught this.

DESIGN. Fire N visually unmistakable images CONCURRENTLY, each with its own colour/shape,
and check that every reply describes the image that request actually sent. Then repeat
SEQUENTIALLY as a control — sequential is the known-good path, so if sequential passes and
concurrent fails, batching is implicated rather than the images or the prompt.

    docker exec longcat-next python3 /workspace/scripts/probe_mm_concurrency.py
Env: CONC_N (default 3), CONC_ROUNDS (default 2).
"""
import base64, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

BASE = "http://localhost:%s" % os.environ.get("PORT", "8090")
KEY = os.environ.get("LCN_API_KEY", "")
N = int(os.environ.get("CONC_N", "3"))
ROUNDS = int(os.environ.get("CONC_ROUNDS", "2"))

# (label, BGR fill, shape, keywords that must appear in a correct description)
SPECS = [
    ("red/circle",   (40, 40, 200),  "circle", ("circle", "round"),  ("red",)),
    ("green/square", (60, 180, 60),  "square", ("square", "rectang"), ("green",)),
    ("blue/triangle", (200, 80, 40), "triangle", ("triangle",),      ("blue",)),
]


def make(spec, variant=0):
    """`variant` resizes the shape by a few pixels.

    The point is a DIFFERENT HASH (so the request is cold) with an IDENTICAL DESCRIPTION.
    Re-colouring would also change the hash, but it changes the right answer too — a
    tinted "red" becomes arguably orange, the colour keyword check fails, and a false BAD
    is indistinguishable from real corruption. Size is the safe knob: nobody describes a
    circle differently because it is 6px larger.
    """
    _, fill, shape, _, _ = spec
    v = int(variant)
    img = np.zeros((512, 512, 3), np.uint8)
    img[:, :] = fill
    if shape == "circle":
        cv2.circle(img, (256, 256), 150 + v, (255, 255, 255), -1)
    elif shape == "square":
        cv2.rectangle(img, (120 - v, 120 - v), (390 + v, 390 + v), (0, 0, 0), -1)
    else:
        cv2.fillPoly(img, [np.array([[256, 110 - v], [400 + v, 390], [112 - v, 390]])],
                     (255, 255, 255))
    return cv2.imencode(".png", img)[1].tobytes()


def ask(png, prompt="Describe this image in one short sentence."):
    b = base64.b64encode(png).decode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps({"model": "longcat-next", "max_tokens": 40, "temperature": 0,
                         "messages": [{"role": "user", "content": [
                             {"type": "text", "text": prompt},
                             {"type": "image_url",
                              "image_url": {"url": "data:image/png;base64," + b}}]}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read())
    return (j["choices"][0]["message"]["content"] or "").strip().replace("\n", " ")


def check(spec, text):
    """Correct = the reply mentions this image's shape AND its colour."""
    _, _, _, shape_kw, colour_kw = spec
    t = text.lower()
    return any(k in t for k in shape_kw) and any(k in t for k in colour_kw)


def run(specs, images, concurrent):
    if concurrent:
        with ThreadPoolExecutor(max_workers=len(images)) as ex:
            texts = list(ex.map(ask, images))
    else:
        texts = [ask(i) for i in images]
    ok = 0
    for spec, text in zip(specs, texts):
        good = check(spec, text)
        ok += good
        print(f"    [{'ok ' if good else 'BAD'}] {spec[0]:14s} -> {text[:80]!r}", flush=True)
    return ok


def main():
    """Each arm gets its OWN fresh image set.

    This matters more than it looks. The first version of this probe ran CONCURRENT then
    SEQUENTIAL over the same images and prompt: the concurrent round corrupted the answers
    AND cached them, so the sequential "control" replayed the cached corruption and both
    arms scored alike — which reads as "concurrency is innocent" and is exactly wrong.
    Reversing the order just moves the confound (a correct cache makes the concurrent arm
    a pure cache hit that never prefills). Disjoint image sets are the only arrangement
    where both arms are genuinely cold and neither can see the other's cache entries.
    """
    specs = SPECS[:N]
    print(f"probe_mm_concurrency: {N} images per arm, disjoint variants, {ROUNDS} round(s)\n",
          flush=True)

    conc_ok = conc_tot = seq_ok = seq_tot = 0
    v = 0
    for r in range(1, ROUNDS + 1):
        # Distinct variant per arm per round => every request below is a cold prefill,
        # and no arm can ever hit a cache entry the other arm created.
        seq_imgs = [make(s, v) for s in specs]; v += 3
        conc_imgs = [make(s, v) for s in specs]; v += 3
        print(f"round {r}: SEQUENTIAL (cold, own variants)", flush=True)
        seq_ok += run(specs, seq_imgs, False); seq_tot += len(specs)
        print(f"round {r}: CONCURRENT (cold, DIFFERENT variants)", flush=True)
        conc_ok += run(specs, conc_imgs, True); conc_tot += len(specs)

    print(f"\nRESULT  sequential {seq_ok}/{seq_tot}   concurrent {conc_ok}/{conc_tot}", flush=True)
    print("READ: sequential passing while concurrent fails implicates batched-prefill "
          "offset handling. Both failing means the images or the check are at fault, "
          "not concurrency.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
