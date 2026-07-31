#!/usr/bin/env python3
"""Extract the codebook embedding sidecar the serving overlay requires.

The serving engine loads text embeddings through SGLang's VocabParallelEmbedding
(text vocab only). The depth-transformer generation heads additionally need the
multimodal codebook rows of the original `model.embed_tokens.weight`
[vocab_size, hidden] (text + audio codebooks + visual codebooks). Those rows are
served from a sidecar file `codebook_embeddings.safetensors` in the model
directory. WITHOUT it the overlay silently falls back to ZERO vectors for all
prior-level codebook conditioning, degrading both audio and image generation
(every codebook level beyond level 0 samples blind to its siblings).

The quantization recipe keeps embed_tokens unquantized, so the full table ships
in the w8a8 shards — this script just slices it out. Run once per deployment:

    python3 extract_codebook_embeddings.py /path/to/LongCat-Next-w8a8int8
"""
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

CODEBOOK_BASE_KEY = "text_vocab_plus_multimodal_special_token_size"  # 131125
EMBED_KEY = "model.embed_tokens.weight"
SIDECAR = "codebook_embeddings.safetensors"


def main(model_dir: str) -> None:
    out_path = os.path.join(model_dir, SIDECAR)
    if os.path.exists(out_path):
        print(f"{out_path} already exists — nothing to do")
        return

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    codebook_base = config.get(CODEBOOK_BASE_KEY, 131125)

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shard = index["weight_map"][EMBED_KEY]

    with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
        full = f.get_slice(EMBED_KEY)
        vocab, hidden = full.get_shape()
        codebook_rows = full[codebook_base:]
    print(f"embed_tokens [{vocab}, {hidden}] -> codebook rows "
          f"[{vocab - codebook_base}, {hidden}] (rows {codebook_base}:)")

    save_file({"codebook_embeddings": codebook_rows.contiguous()}, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
