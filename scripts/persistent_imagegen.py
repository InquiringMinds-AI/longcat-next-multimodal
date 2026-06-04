#!/usr/bin/env python3
"""
Persistent IMAGE generation driver: load the LongCat-Next backbone + gen modules
ONCE, then loop over a CONFIGS list [{prompt,cfg,tag}], generating each with a
fresh KV/req-pool reset (same pattern as persistent_audiogen.py). Saves
/tmp/gen_ids_<tag>.pt per config. Reuses helpers from gen_image_standalone.
Env: CONFIGS (json list), MODEL_PATH.
"""
import os, json, torch
import gen_image_standalone as G   # __main__-guarded, so import does not run main()

log = G.log
IMG_START, IMG_PAD, IMG_NEWLINE = G.IMG_START, G.IMG_PAD, G.IMG_NEWLINE
NUM_CB, HIDDEN, TOKENS_H, TOKENS_W = G.NUM_CB, G.HIDDEN, G.TOKENS_H, G.TOKENS_W
NVFP4 = G.NVFP4


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

    override = {
        "architectures": ["LongcatFlashForCausalLM"],
        "use_ngram_embedding": True,
        "ngram_embedding_m": 10223616, "ngram_embedding_n": 5, "ngram_embedding_k": 3,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
        "disable_quant_module": ["self_attn"],
    }
    server_args = ServerArgs(
        model_path=NVFP4, tokenizer_path=NVFP4, quantization="modelopt_fp4",
        json_model_override_args=json.dumps(override), attention_backend="flashinfer",
        mem_fraction_static=0.5, max_total_tokens=4096, dtype="bfloat16",
        disable_cuda_graph=True, disable_radix_cache=True, skip_server_warmup=True,
        trust_remote_code=True, context_length=4096, chunked_prefill_size=4096, device="cuda",
    )
    log("[P.runner] building ModelConfig + ModelRunner ...")
    model_config = ModelConfig.from_server_args(server_args)
    port_args = PortArgs.init_new(server_args)
    mr = ModelRunner(model_config=model_config, mem_fraction_static=0.5, gpu_id=0,
                     tp_rank=0, tp_size=1, moe_ep_rank=0, moe_ep_size=1, pp_rank=0, pp_size=1,
                     nccl_port=port_args.nccl_port, server_args=server_args)
    tok = get_tokenizer(NVFP4, trust_remote_code=True)
    bridge, head, codebook_sizes, mm_embed_rows = G.build_gen_modules()
    model = mr.model
    log("=== P.MILESTONE: ModelRunner + gen modules loaded ===")

    NE_N = mr.model_config.hf_config.ngram_embedding_n
    token_table = mr.token_table
    tv = getattr(mr.model_config.hf_config, "text_vocab_size", 131072)
    tvp = getattr(mr.model_config.hf_config, "text_vocab_plus_multimodal_special_token_size", 131125)

    def _zero_mm(t):
        m = (t >= tv) & (t < tvp)
        if m.any():
            t = t.clone(); t[m] = 0
        return t

    class _TC(SimpleNamespace):
        def supports_swa(self): return False
        def supports_mamba(self): return False
        def is_chunk_cache(self): return False
        def is_tree_cache(self): return True
        def evict(self, *a, **k): pass

    def reset_pools():
        for _alloc in (mr.token_to_kv_pool_allocator, mr.req_to_token_pool):
            for _m in ('clear', 'reset', 'free_all'):
                if hasattr(_alloc, _m):
                    try: getattr(_alloc, _m)()
                    except Exception as e: log(f"[reset] {_alloc.__class__.__name__}.{_m} -> {e}")
                    break

    SUFFIX = "<longcat_img_token_size>18 18</longcat_img_token_size><longcat_img_start>"

    @torch.no_grad()
    def gen_once(prompt_text, cfg_scale, tag, neg=""):
        reset_pools()
        cond = prompt_text.rstrip() + " " + SUFFIX
        uncond = (neg.rstrip() + " " + SUFFIX) if neg else SUFFIX
        reqs = []
        for ri, ptext in enumerate([cond, uncond]):
            pids = tok.encode(ptext)
            spp = SamplingParams(temperature=0, max_new_tokens=400); spp.normalize(tok)
            rq = Req(rid=ri, origin_input_text=ptext, origin_input_ids=list(pids), sampling_params=spp)
            rq.fill_ids = list(pids); rq.logprob_start_len = -1
            rq.set_extend_input_len(len(rq.fill_ids) - len(rq.prefix_indices))
            reqs.append(rq)
        BS = len(reqs)
        dummy_tc = _TC(page_size=server_args.page_size, device=mr.device,
                       token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator)
        batch = ScheduleBatch.init_new(
            reqs=reqs, req_to_token_pool=mr.req_to_token_pool,
            token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
            tree_cache=dummy_tc, model_config=mr.model_config,
            enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE,
        )

        def prefill_token_table():
            batch.ne_token_table = token_table
            all_tokens, column_starts, req_lens = [], [], []
            for rq in batch.reqs:
                start = len(rq.prefix_indices); end = start + rq.extend_input_len
                fill_ids = rq.origin_input_ids + rq.output_ids
                if start == 0: toks = fill_ids[start:end]; cs = 0
                elif start < NE_N: toks = fill_ids[0:end]; cs = 0
                else: toks = fill_ids[start - NE_N + 1:end]; cs = start - NE_N + 1
                all_tokens.extend(toks); column_starts.append(cs); req_lens.append(len(toks))
            dev = token_table.device
            tt = _zero_mm(torch.tensor(all_tokens, dtype=token_table.dtype, device=dev))
            update_token_table(ne_token_table=token_table, tokens=tt, row_indices=batch.req_pool_indices,
                               column_starts=torch.tensor(column_starts, dtype=torch.int32, device=dev),
                               req_lens=torch.tensor(req_lens, dtype=torch.int32, device=dev), ignore_tokens=None)

        def append_token_table(fb, next_tok_id):
            dev = token_table.device; nrows = fb.req_pool_indices.shape[0]
            tt = _zero_mm(torch.full((nrows,), next_tok_id, dtype=torch.int32, device=dev))
            update_token_table(ne_token_table=token_table, tokens=tt, row_indices=fb.req_pool_indices,
                               column_starts=fb.seq_lens.to(torch.int32),
                               req_lens=torch.ones_like(fb.seq_lens, dtype=torch.int32), ignore_tokens=None)

        def run_forward(input_embeds_override=None):
            mwb = batch.get_model_worker_batch(); mwb.ne_token_table = token_table
            fb = ForwardBatch.init_new(mwb, mr); fb.capture_hidden_mode = CaptureHiddenMode.LAST
            mr.attn_backend.init_forward_metadata(fb)
            if input_embeds_override is not None:
                lo = model.forward(None, fb.positions, fb, input_embeds=input_embeds_override)
            else:
                lo = model.forward(fb.input_ids, fb.positions, fb, input_embeds=None)
            return lo, fb

        def split_hidden(lo):
            h = lo.hidden_states.reshape(BS, HIDDEN).to(torch.bfloat16)
            return h[0:1], (h[1:2] if BS == 2 else None)

        def depth_step(hc, hu):
            if BS == 2:
                return G.gen_codebooks_cfg(hc, hu, bridge, head, codebook_sizes, cfg_scale)
            return G.gen_codebooks(hc, bridge, head, codebook_sizes)

        def feed_for(ids):
            fe = G.feedback_embed(ids, bridge, mm_embed_rows)
            return fe.expand(BS, HIDDEN).contiguous()

        batch.prepare_for_extend(); prefill_token_table()
        lo, fb = run_forward()
        hc, hu = split_hidden(lo)
        cur_ids = depth_step(hc, hu)
        all_ids = []; pos_count = 0
        for row in range(TOKENS_H):
            for col in range(TOKENS_W):
                all_ids.append(cur_ids.reshape(NUM_CB).clone()); pos_count += 1
                feed_emb = feed_for(cur_ids)
                batch.output_ids = torch.full((BS,), IMG_PAD, dtype=torch.long, device=mr.device)
                batch.prepare_for_decode()
                lo, fbi = run_forward(input_embeds_override=feed_emb)
                append_token_table(fbi, IMG_PAD)
                hc, hu = split_hidden(lo)
                if pos_count < TOKENS_H * TOKENS_W: cur_ids = depth_step(hc, hu)
            batch.output_ids = torch.full((BS,), IMG_NEWLINE, dtype=torch.long, device=mr.device)
            batch.prepare_for_decode()
            lo, fbi = run_forward(input_embeds_override=None)
            append_token_table(fbi, IMG_NEWLINE)
            hc, hu = split_hidden(lo)
            if pos_count < TOKENS_H * TOKENS_W: cur_ids = depth_step(hc, hu)
        gen_ids = torch.stack(all_ids, dim=0).cpu()
        l0 = torch.unique(gen_ids[:, 0]).numel()
        out = f"/tmp/gen_ids_{tag}.pt"; torch.save(gen_ids, out)
        log(f"=== DONE {tag}: gen_ids{tuple(gen_ids.shape)} level0_unique={l0} -> {out} ===")

    CONFIGS = json.load(open(os.environ["CONFIGS_FILE"]))
    for c in CONFIGS:
        if "top_k" in c: G.TOP_K = int(c["top_k"])
        if "temp" in c: G.TEMP = float(c["temp"])
        if "top_p" in c: G.TOP_P = float(c["top_p"])
        G.TOP_K_LEVELS = list(c["topk_levels"]) if "topk_levels" in c else None
        log(f"=== GEN tag={c['tag']} cfg={c['cfg']} temp={G.TEMP} top_p={G.TOP_P} top_k={G.TOP_K} topk_levels={G.TOP_K_LEVELS} prompt={c['prompt']!r} ===")
        try:
            gen_once(c["prompt"], float(c["cfg"]), c["tag"], c.get("neg", ""))
        except Exception as e:
            import traceback; log(f"[ERR {c['tag']}] {e}"); traceback.print_exc()
    log("ALL CONFIGS DONE")


if __name__ == "__main__":
    main()
