"""Generate forward golden files from a dump_init file (ROADMAP §10, P2.6).

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
logits f64 [B, T, V]. P3.7 extends this script with gradients.
"""

import array
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


def forward(d):
    V, maxT, L, C, NH, tied, B, T = d["config"][2]
    hs = C // NH
    p = lambda name: tensor(d["param." + name])
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


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: gen_golden.py <init.bin> <out_fwd.bin>")
    torch.set_default_dtype(torch.float64)
    d = read_tsgd(sys.argv[1])
    V, maxT, L, C, NH, tied, B, T = d["config"][2]
    with torch.no_grad():
        logits, losses = forward(d)
    loss = losses.mean()
    write_tsgd(sys.argv[2], [
        ("loss", [1], [loss.item()]),
        ("losses", [B * T], losses.tolist()),
        ("logits", [B, T, V], logits.reshape(-1).tolist()),
    ])
    print("%s: loss %.12f (torch %s)" % (sys.argv[2], loss.item(), torch.__version__))


if __name__ == "__main__":
    main()
