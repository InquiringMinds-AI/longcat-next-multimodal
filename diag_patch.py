p = "/home/magi/scripts/stream_calibrate.py"
s = open(p).read()
anchor = '    print(f"[verify] backbone meta-leftover tensors: {len(leftover)}", flush=True)\n'
diag = anchor + (
    "    _allmeta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == 'meta']\n"
    "    print(f'[diag] ALL meta tensors (incl stubbed): {len(_allmeta)}', flush=True)\n"
    "    for _n in _allmeta[:40]: print('    META:', _n, flush=True)\n"
)
assert anchor in s
s = s.replace(anchor, diag, 1)
# also relax the raise so it does not abort before we run forward diagnosis... keep raise but only on non-inv_freq
open(p, "w").write(s)
print("diag added")
