"""Generate forward and backward golden files from a dump_init file (ROADMAP §10, P2.6, P3.7).

OFFLINE / DEV ONLY. Never run by make or any test. Its outputs are committed as
plain data under tests/golden/ and read by tests/test_golden.cpp.

Usage (from the repo root, PyTorch in a dev venv, see dev_only/requirements.txt):
  bin/dump_init --out tests/golden/tiny_init.bin ...      (see tests/golden/README.md)
  python dev_only/gen_golden.py tests/golden/tiny_init.bin tests/golden/tiny_fwd.bin

The model below is an architecturally identical float64 PyTorch rewrite of
src/model/model.cpp, written from ROADMAP §2.1/§6:
  x = wte[tokens] + wpe[:T]
  per layer: x += proj(causal_attn(qkv(LN1(x))));  x += fcproj(gelu_tanh(fc(LN2(x))))
  logits = LNf(x) @ head_w^T   (head_w = wte when tied; no head bias)
  loss = mean cross-entropy
LayerNorm eps 1e-5 (biased variance), GELU tanh approximation, attention scale
1/sqrt(hs), linear weights [OC, C]. Parameters are the f32 values from the dump,
widened to float64.

Output (tensor_file.h "TSGD" format): loss f64 [1], losses f64 [B*T],
logits f64 [B, T, V], and grad.<name> f64 [shape] for every parameter: the
autograd gradient of the mean loss (P3.7).
Optional third output (P4.1): a 5-step training trajectory on the same
fixed batch (global-norm clip, then torch.optim.AdamW with decay/no-decay
groups, warmup+cosine LR): hparams, step_loss, step_grad_norm (pre-clip),
step_lr, and the final param.<name> values.
"""

import array
import math
import struct
import sys

import torch
import torch.nn.functional as F

DTYPES = {0: ("f", 4), 1: ("d", 8), 2: ("i", 4), 3: ("q", 8)}


def read_tsgd(path):
    assert sys.byteorder == "little", "this reader assumes a little-endian host"
    b = open(path, "rb").read()
    assert b[:4] == b"TSGD", "bad magic"
    version, count = struct.unpack_from("<II", b, 4)
    assert version == 1
    p, out = 12, {}
    for _ in range(count):
        (nl,) = struct.unpack_from("<I", b, p); p += 4
        name = b[p:p + nl].decode(); p += nl
        dt, nd = struct.unpack_from("<II", b, p); p += 8
        dims = list(struct.unpack_from("<%dQ" % nd, b, p)); p += 8 * nd
        code, size = DTYPES[dt]
        n = 1
        for d in dims:
            n *= d
        a = array.array(code)
        a.frombytes(b[p:p + n * size]); p += n * size
        out[name] = (dt, dims, list(a))
    assert p == len(b), "trailing bytes"
    return out


def write_tsgd(path, tensors):
    """tensors: list of (name, dims, list of float) written as f64."""
    out = bytearray(b"TSGD") + struct.pack("<II", 1, len(tensors))
    for name, dims, vals in tensors:
        nb = name.encode()
        out += struct.pack("<I", len(nb)) + nb + struct.pack("<II", 1, len(dims))
        out += struct.pack("<%dQ" % len(dims), *dims)
        out += struct.pack("<%dd" % len(vals), *vals)
    open(path, "wb").write(bytes(out))


def tensor(entry):
    _, dims, vals = entry
    return torch.tensor(vals, dtype=torch.float64).reshape(dims)


def forward(d, P):
    """P: parameter name -> float64 tensor (leaf tensors when gradients are wanted)."""
    V, maxT, L, C, NH, tied, B, T = d["config"][2]
    hs = C // NH
    p = lambda name: P[name]
    tokens = torch.tensor(d["tokens"][2], dtype=torch.long).reshape(B, T)
    targets = torch.tensor(d["targets"][2], dtype=torch.long).reshape(B, T)

    wte = p("wte")
    x = wte[tokens] + p("wpe")[:T]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for l in range(L):
        q = lambda n: p("h%d.%s" % (l, n))
        h = F.layer_norm(x, (C,), q("ln1_g"), q("ln1_b"), eps=1e-5)
        qkv = h @ q("qkv_w").T + q("qkv_b")
        Q, K, Vv = qkv.split(C, dim=-1)
        Q, K, Vv = (t.reshape(B, T, NH, hs).transpose(1, 2) for t in (Q, K, Vv))
        s = (Q @ K.transpose(-1, -2)) / (hs ** 0.5)
        s = s.masked_fill(~mask, float("-inf"))
        y = torch.softmax(s, dim=-1) @ Vv
        y = y.transpose(1, 2).reshape(B, T, C)
        x = x + (y @ q("proj_w").T + q("proj_b"))
        h = F.layer_norm(x, (C,), q("ln2_g"), q("ln2_b"), eps=1e-5)
        h = F.gelu(h @ q("fc_w").T + q("fc_b"), approximate="tanh")
        x = x + (h @ q("fcproj_w").T + q("fcproj_b"))
    x = F.layer_norm(x, (C,), p("lnf_g"), p("lnf_b"), eps=1e-5)
    head = wte if tied else p("head_w")
    logits = x @ head.T
    losses = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none")
    return logits, losses


# AdamW trajectory hyperparameters (P4.1). Stored in the output file so the
# C++ test reproduces exactly this run.
ADAMW = dict(lr=1e-3, min_lr_frac=0.1, warmup=2, max_steps=5, beta1=0.9, beta2=0.95, eps=1e-8,
             weight_decay=0.1, clip=1.0, steps=5)


def lr_at(step, h):
    """ROADMAP 2.3 schedule, written from the formula: linear warmup, then cosine to min_lr_frac*lr."""
    peak, floor_lr, W, S = h["lr"], h["min_lr_frac"] * h["lr"], h["warmup"], h["max_steps"]
    if W > 0 and step <= W:
        return peak * step / W
    q = min(1.0, (step - W) / (S - W))
    return floor_lr + 0.5 * (peak - floor_lr) * (1 + math.cos(math.pi * q))


def adamw_trajectory(d, names, h):
    """`steps` training steps on the fixed batch: grads -> global-norm clip -> AdamW.
    Clip: scale by clip/norm only if norm > clip (no epsilon, unlike
    torch.nn.utils.clip_grad_norm_). Weight decay only on 2-D weights except wpe."""
    P = {n: tensor(d["param." + n]).requires_grad_(True) for n in names}
    decay = [n for n in names if P[n].dim() == 2 and n != "wpe"]
    no_decay = [n for n in names if n not in decay]
    opt = torch.optim.AdamW(
        [{"params": [P[n] for n in decay], "weight_decay": h["weight_decay"]},
         {"params": [P[n] for n in no_decay], "weight_decay": 0.0}],
        lr=h["lr"], betas=(h["beta1"], h["beta2"]), eps=h["eps"])
    losses, norms, lrs = [], [], []
    for step in range(1, h["steps"] + 1):
        opt.zero_grad(set_to_none=False)
        _, l = forward(d, P)
        loss = l.mean()
        loss.backward()
        norm = math.sqrt(sum(float((P[n].grad ** 2).sum()) for n in names))
        if norm > h["clip"]:
            with torch.no_grad():
                for n in names:
                    P[n].grad.mul_(h["clip"] / norm)
        lr = lr_at(step, h)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        losses.append(loss.item()); norms.append(norm); lrs.append(lr)
    return P, decay, losses, norms, lrs


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: gen_golden.py <init.bin> <out_fwd.bin> [<out_adamw.bin>]")
    torch.set_default_dtype(torch.float64)
    d = read_tsgd(sys.argv[1])
    V, maxT, L, C, NH, tied, B, T = d["config"][2]
    # Every parameter as a float64 leaf tensor, in the dump (= layout) order.
    names = [k[len("param."):] for k in d if k.startswith("param.")]
    P = {n: tensor(d["param." + n]).requires_grad_(True) for n in names}
    logits, losses = forward(d, P)
    loss = losses.mean()
    loss.backward()  # autograd: d(mean loss)/d(param); tied wte gets both paths
    out = [
        ("loss", [1], [loss.item()]),
        ("losses", [B * T], losses.detach().tolist()),
        ("logits", [B, T, V], logits.detach().reshape(-1).tolist()),
    ]
    for n in names:
        g = P[n].grad
        out.append(("grad." + n, list(g.shape), g.reshape(-1).tolist()))
    write_tsgd(sys.argv[2], out)
    print("%s: loss %.12f, %d gradient tensors (torch %s)" % (sys.argv[2], loss.item(), len(names), torch.__version__))
    if len(sys.argv) == 4:
        h = ADAMW
        Pt, decay, step_losses, norms, lrs = adamw_trajectory(d, names, h)
        hp = [h["lr"], h["min_lr_frac"], h["warmup"], h["max_steps"], h["beta1"], h["beta2"], h["eps"],
              h["weight_decay"], h["clip"], h["steps"]]
        out2 = [
            ("hparams", [len(hp)], hp),  # lr, min_lr_frac, warmup, max_steps, b1, b2, eps, wd, clip, steps
            ("step_loss", [h["steps"]], step_losses),
            ("step_grad_norm", [h["steps"]], norms),
            ("step_lr", [h["steps"]], lrs),
        ]
        for n in names:
            out2.append(("param." + n, list(Pt[n].shape), Pt[n].detach().reshape(-1).tolist()))
        write_tsgd(sys.argv[3], out2)
        print("%s: %d AdamW steps, losses %s, decay tensors %d" % (
            sys.argv[3], h["steps"], " ".join("%.6f" % x for x in step_losses), len(decay)))


if __name__ == "__main__":
    main()
