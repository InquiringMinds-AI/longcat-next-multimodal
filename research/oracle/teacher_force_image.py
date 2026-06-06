#!/usr/bin/env python3
"""TEACHER-FORCING diagnostic for image gen. Feeds a REAL image's [324,8] codes
through the generation loop as history (no sampling of own tokens), and at each
position measures whether the depth head's argmax matches the REAL next token.
High level-0 top-1 => backbone hidden CAN drive correct gen (failure = AR drift);
near-chance => NVFP4 hidden can't carry correct content (precision wall).
Saves argmax tokens /tmp/gen_ids_tf_<name>.pt for optional decode (user judges).
Env: REAL_CODES (default /tmp/real_codes.pt), MODEL_PATH."""
import os, json, torch
import gen_image_standalone as G

log = G.log
IMG_START, IMG_PAD, IMG_NEWLINE = G.IMG_START, G.IMG_PAD, G.IMG_NEWLINE
NUM_CB, HIDDEN, TOKENS_H, TOKENS_W = G.NUM_CB, G.HIDDEN, G.TOKENS_H, G.TOKENS_W
NVFP4 = G.NVFP4
SUFFIX = "<longcat_img_token_size>18 18</longcat_img_token_size><longcat_img_start>"
PROMPTS = {
    "red_circle": "A single large red circle centered on a plain white background.",
    "food0": "A vivid color photograph of a colorful dish of food.",
    "food1": "A vivid color photograph of a colorful dish of food.",
}


def main():
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs, PortArgs
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, CaptureHiddenMode
    from sglang.srt.managers.schedule_batch import ScheduleBatch, Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer
    from sglang.jit_kernel.ngram_embedding import update_token_table
    from types import SimpleNamespace

    override = {"architectures": ["LongcatFlashForCausalLM"], "use_ngram_embedding": True,
                "ngram_embedding_m": 10223616, "ngram_embedding_n": 5, "ngram_embedding_k": 3,
                "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
                "disable_quant_module": ["self_attn"]}
    server_args = ServerArgs(model_path=NVFP4, tokenizer_path=NVFP4, quantization="modelopt_fp4",
        json_model_override_args=json.dumps(override), attention_backend="flashinfer",
        mem_fraction_static=0.5, max_total_tokens=4096, dtype="bfloat16", disable_cuda_graph=True,
        disable_radix_cache=True, skip_server_warmup=True, trust_remote_code=True,
        context_length=4096, chunked_prefill_size=4096, device="cuda")
    mc = ModelConfig.from_server_args(server_args); pa = PortArgs.init_new(server_args)
    mr = ModelRunner(model_config=mc, mem_fraction_static=0.5, gpu_id=0, tp_rank=0, tp_size=1,
                     moe_ep_rank=0, moe_ep_size=1, pp_rank=0, pp_size=1, nccl_port=pa.nccl_port, server_args=server_args)
    tok = get_tokenizer(NVFP4, trust_remote_code=True)
    bridge, head, codebook_sizes, mm_embed_rows = G.build_gen_modules()
    model = mr.model
    log("=== TF.loaded ===")

    NE_N = mr.model_config.hf_config.ngram_embedding_n
    token_table = mr.token_table
    tv = getattr(mr.model_config.hf_config, "text_vocab_size", 131072)
    tvp = getattr(mr.model_config.hf_config, "text_vocab_plus_multimodal_special_token_size", 131125)
    def _zero_mm(t):
        m = (t >= tv) & (t < tvp)
        if m.any(): t = t.clone(); t[m] = 0
        return t
    class _TC(SimpleNamespace):
        def supports_swa(self): return False
        def supports_mamba(self): return False
        def is_chunk_cache(self): return False
        def is_tree_cache(self): return True
        def evict(self, *a, **k): pass
    def reset_pools():
        for al in (mr.token_to_kv_pool_allocator, mr.req_to_token_pool):
            for m in ('clear', 'reset', 'free_all'):
                if hasattr(al, m):
                    try: getattr(al, m)()
                    except Exception: pass
                    break

    @torch.no_grad()
    def head_logits(hidden, real_row, i):
        # real_row: LongTensor[8] of REAL codes; head sees real previous levels
        next_ids = real_row.view(1, NUM_CB).to(hidden.device)
        logits = head(hidden, next_ids, bridge.embedding_layers, 1, i)  # [1, CB+1]
        logits = logits.clone()
        logits[:, codebook_sizes[i]] = torch.finfo(logits.dtype).min  # mask EOS slot
        return logits[0]

    @torch.no_grad()
    def teacher_force(name, real_codes):
        reset_pools()
        prompt = PROMPTS.get(name, "A photograph.")
        ptext = prompt.rstrip() + " " + SUFFIX
        pids = tok.encode(ptext)
        spp = SamplingParams(temperature=0, max_new_tokens=400); spp.normalize(tok)
        rq = Req(rid=0, origin_input_text=ptext, origin_input_ids=list(pids), sampling_params=spp)
        rq.fill_ids = list(pids); rq.logprob_start_len = -1
        rq.set_extend_input_len(len(rq.fill_ids) - len(rq.prefix_indices))
        dummy_tc = _TC(page_size=server_args.page_size, device=mr.device,
                       token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator)
        batch = ScheduleBatch.init_new(reqs=[rq], req_to_token_pool=mr.req_to_token_pool,
            token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator, tree_cache=dummy_tc,
            model_config=mr.model_config, enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)

        def prefill_tt():
            batch.ne_token_table = token_table
            allt, cs_l, rl_l = [], [], []
            for r in batch.reqs:
                start = len(r.prefix_indices); end = start + r.extend_input_len
                fi = r.origin_input_ids + r.output_ids
                if start == 0: toks = fi[start:end]; cs = 0
                elif start < NE_N: toks = fi[0:end]; cs = 0
                else: toks = fi[start - NE_N + 1:end]; cs = start - NE_N + 1
                allt.extend(toks); cs_l.append(cs); rl_l.append(len(toks))
            dev = token_table.device
            update_token_table(ne_token_table=token_table, tokens=_zero_mm(torch.tensor(allt, dtype=token_table.dtype, device=dev)),
                row_indices=batch.req_pool_indices, column_starts=torch.tensor(cs_l, dtype=torch.int32, device=dev),
                req_lens=torch.tensor(rl_l, dtype=torch.int32, device=dev), ignore_tokens=None)

        def append_tt(fb, tid):
            dev = token_table.device; n = fb.req_pool_indices.shape[0]
            update_token_table(ne_token_table=token_table, tokens=_zero_mm(torch.full((n,), tid, dtype=torch.int32, device=dev)),
                row_indices=fb.req_pool_indices, column_starts=fb.seq_lens.to(torch.int32),
                req_lens=torch.ones_like(fb.seq_lens, dtype=torch.int32), ignore_tokens=None)

        def run_fwd(emb=None):
            mwb = batch.get_model_worker_batch(); mwb.ne_token_table = token_table
            fb = ForwardBatch.init_new(mwb, mr); fb.capture_hidden_mode = CaptureHiddenMode.LAST
            mr.attn_backend.init_forward_metadata(fb)
            lo = model.forward(None if emb is not None else fb.input_ids, fb.positions, fb, input_embeds=emb)
            return lo, fb
        def hid(lo): return lo.hidden_states.reshape(1, HIDDEN).to(torch.bfloat16)

        batch.prepare_for_extend(); prefill_tt()
        lo, fb = run_fwd(); h = hid(lo)
        real = real_codes.to(mr.device)
        hits = torch.zeros(NUM_CB); ranks = [[] for _ in range(NUM_CB)]
        argmax_ids = torch.zeros(324, NUM_CB, dtype=torch.long)
        pos = 0
        for rrow in range(TOKENS_H):
            for ccol in range(TOKENS_W):
                rr = real[pos]
                for i in range(NUM_CB):
                    lg = head_logits(h, rr, i)
                    pred = int(lg.argmax()); tgt = int(rr[i])
                    argmax_ids[pos, i] = pred
                    if pred == tgt: hits[i] += 1
                    ranks[i].append(int((lg > lg[tgt]).sum()))
                pos += 1
                feed = G.feedback_embed(rr.view(1, NUM_CB), bridge, mm_embed_rows).expand(1, HIDDEN).contiguous()
                batch.output_ids = torch.full((1,), IMG_PAD, dtype=torch.long, device=mr.device)
                batch.prepare_for_decode()
                lo, fbi = run_fwd(emb=feed); append_tt(fbi, IMG_PAD); h = hid(lo)
            batch.output_ids = torch.full((1,), IMG_NEWLINE, dtype=torch.long, device=mr.device)
            batch.prepare_for_decode()
            lo, fbi = run_fwd(emb=None); append_tt(fbi, IMG_NEWLINE); h = hid(lo)
        torch.save(argmax_ids.cpu(), f"/tmp/gen_ids_tf_{name}.pt")
        acc = (hits / 324.0).tolist()
        medrank = [int(torch.tensor(r).median()) for r in ranks]
        log(f"=== TF {name} (prompt={prompt!r}) ===")
        log(f"    per-level top1 acc: " + " ".join(f"L{i}={acc[i]*100:.1f}%" for i in range(NUM_CB)))
        log(f"    per-level real-token MEDIAN rank (0=top; vocab {codebook_sizes}): " + " ".join(f"L{i}={medrank[i]}" for i in range(NUM_CB)))
        log(f"    -> saved argmax /tmp/gen_ids_tf_{name}.pt")

    real_codes = torch.load(os.environ.get("REAL_CODES", "/tmp/real_codes.pt"), map_location="cpu")
    for name, codes in real_codes.items():
        try: teacher_force(name, codes)
        except Exception as e:
            import traceback; log(f"[ERR {name}] {e}"); traceback.print_exc()
    log("TEACHER FORCE DONE")


if __name__ == "__main__":
    main()
