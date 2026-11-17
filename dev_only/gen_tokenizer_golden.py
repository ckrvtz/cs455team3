"""Generate tokenizer golden files (ROADMAP §7.2, P1.2). OFFLINE / DEV ONLY.

Never run by make or any test. Its outputs are committed as plain data:
  tests/golden/tokenizer_cases.txt          text -> expected ids and pre-token splits (HF)
  tests/golden/tokenizer_vocab_subset.json  vocab entries needed by those cases
  tests/golden/tokenizer_merges_subset.txt  merges applied by those cases, original order

Usage (from the repo root, after scripts/download_data.sh):
  python3 -m venv /tmp/venv && /tmp/venv/bin/pip install -r dev_only/requirements.txt
  /tmp/venv/bin/python dev_only/gen_tokenizer_golden.py

Expected ids come from the Hugging Face `tokenizers` ByteLevel BPE (same
pre-tokenizer and merges as GPT-Neo; no special-token handling).

Why the subset fixture is exact: BPE repeatedly applies the lowest-rank
merge among adjacent pairs. Every merge the full table would prefer at some
step is, by definition, the one applied at that step. So keeping exactly the
applied merges, in their original relative order, reproduces every step for
these inputs. This script checks that claim: it re-runs a plain BPE on HF's
pre-tokens, compares with HF's ids, and records each applied merge.
"""

import json
import random
import sys

from tokenizers import ByteLevelBPETokenizer

VOCAB, MERGES, VALID = "data/vocab.json", "data/merges.txt", "data/TinyStories-valid.txt"
OUT_CASES = "tests/golden/tokenizer_cases.txt"
OUT_VOCAB = "tests/golden/tokenizer_vocab_subset.json"
OUT_MERGES = "tests/golden/tokenizer_merges_subset.txt"
SEED = 455

HAND_CASES = [
    # empty / whitespace
    "", " ", "  ", "   ", "\n", "\n\n", "\r\n", "\t", " \t ", "a  b", "a   b", "a \t b",
    "trailing   ", "   leading", "x\n\n\ny", "line one\nline two\n", "tab\tseparated\tvalues",
    "a\u000bb\u000cc", "\u001cfs\u001fus",
    # contractions (case-sensitive; mid-word; uppercase not a contraction)
    "I'm", "you're", "we've", "she'll", "he'd", "it's", "don't", "can't", "won't",
    "I'M", "YOU'RE", "IT'S", "rock'n'roll", "'s", "'sup", "''", "' s", "x's'd",
    "o'clock", "the dog's bone", "Tom's mom's car",
    # quotes
    "\"Hello,\" she said.", "'quoted'", "“Curly,” he said.", "‘single’",
    "It’s fine.", "„German“", "«French»",
    # digits and numbers
    "123", "3.14", "1,000,000", "2nd", "21st century", "007", "abc123def", "1234567890" * 5,
    "² and ³", "½ cup", "٣٤٥", "１２３", "Ⅻ",
    "①②", "x²+y²",
    # punctuation runs
    "!!!", "...", "?!", "Wait...what?!", "--", "a--b", "(parens) [brackets] {braces}",
    "#hashtag @mention", "50% off $5", "e-mail", "C++ & C#", "a/b\\c", "~`^|<>",
    # unicode punctuation and spaces
    "—em dash—", "en–dash", "wait…", "non breaking", "em space",
    "ideo　space", "zero​width", "﻿bom", "soft­hyphen", "nel\u0085next",
    "line sep", "para sep", "thin space", "¿Qué?", "¡Hola!",
    "5×3÷2", "€10 £5", "TM™", "№ 5", "25℃",
    "。、", "！？", "→ arrow", "♥ heart",
    # letters (non-ASCII)
    "café", "naïve", "Zürich", "résumé", "ªºµ",
    "Ω ohm Ω", "Αθήνα", "Привет мир",
    "你好，世界", "こんにちは", "カタカナ",
    "안녕하세요", "שָׁלוֹם",
    "مرحبا ١٢", "नमस्ते",
    "สวัสดี", "Łódź", "ﬁne",
    "ＡＢＣ",
    # emoji and symbols
    "\U0001F60A", "Hi \U0001F60A!", "\U0001F44D\U0001F3FD", "\U0001F468‍\U0001F469‍\U0001F467",
    "\U0001F1FA\U0001F1F8", "1️⃣", "❤️", "\U0001F431cat", "\U0001F600\U0001F600\U0001F600",
    # the EOS string is plain text for encode() (no special-token handling)
    "<|endoftext|>", "end<|endoftext|>start",
    # long words
    "a" * 1000, "supercalifragilisticexpialidocious" * 10,
    "".join(random.Random(7).choice("abcdefghijklmnopqrstuvwxyz") for _ in range(400)),
    # typical story text
    "Once upon a time, there was a little girl named Lily.",
    "One day, Tim went to the park. He saw a big, red ball!",
]


def escape(s: str) -> str:
    out = []
    for i, ch in enumerate(s):
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif o < 0x20 or o == 0x7F or (i == 0 and ch == "#"):
            out.append("\\x%02X" % o)
        else:
            out.append(ch)
    return "".join(out)


def bpe_trace(piece, ranks, vocab, used):
    """Plain BPE on one pre-token in byte-alphabet form; records applied merges."""
    syms = list(piece)
    while len(syms) > 1:
        best = None
        for a, b in zip(syms, syms[1:]):
            r = ranks.get((a, b))
            if r is not None and (best is None or r < best[0]):
                best = (r, a, b)
        if best is None:
            break
        r, a, b = best
        used.add(r)
        merged, i = [], 0
        while i < len(syms):
            if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(syms[i])
                i += 1
        syms = merged
    return [vocab[s] for s in syms]


def main():
    tok = ByteLevelBPETokenizer(VOCAB, MERGES)  # add_prefix_space=False
    vocab = json.load(open(VOCAB, encoding="utf-8"))
    merge_lines = [l for l in open(MERGES, encoding="utf-8").read().split("\n")[1:] if l]
    ranks = {tuple(l.split(" ")): i for i, l in enumerate(merge_lines)}

    # Real TinyStories lines: a seeded sample, plus every non-ASCII line (capped).
    lines = [l for l in open(VALID, encoding="utf-8").read().split("\n") if l.strip()]
    rng = random.Random(SEED)
    sample = rng.sample(lines, 130)
    nonascii = [l for l in lines if any(ord(c) > 127 for c in l)]
    sample += rng.sample(nonascii, min(40, len(nonascii)))
    cases = HAND_CASES + sample

    used = set()
    rows = []
    for text in cases:
        ids = tok.encode(text).ids
        pieces = [p for p, _ in tok.pre_tokenizer.pre_tokenize_str(text)]
        traced = [i for p in pieces for i in bpe_trace(p, ranks, vocab, used)]
        if traced != ids:
            sys.exit(f"trace mismatch for {text!r}: {traced} vs {ids}")
        if tok.decode(ids) != text:
            sys.exit(f"HF round trip failed for {text!r}")
        # Pre-token byte lengths (one alphabet char per byte) pin the scanner and
        # its character classes directly; ids alone rarely reveal class errors
        # because GPT-2 merges never cross pre-token boundaries.
        rows.append(escape(text) + "\t" + " ".join(map(str, ids)) + "\t" + " ".join(str(len(p)) for p in pieces))

    used_sorted = sorted(used)
    byte_tokens = [t for t in vocab if len(t) == 1]
    assert len(byte_tokens) == 256, len(byte_tokens)
    keep = set(byte_tokens)
    for r in used_sorted:
        a, b = merge_lines[r].split(" ")
        keep.update((a, b, a + b))
    subset = {t: vocab[t] for t in sorted(keep, key=lambda t: vocab[t])}

    with open(OUT_CASES, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Tokenizer golden cases (ROADMAP §7.2). GENERATED by dev_only/gen_tokenizer_golden.py;\n")
        f.write("# do not edit. Expected ids: HF tokenizers %s ByteLevel BPE, GPT-Neo vocab/merges\n"
                % __import__("tokenizers").__version__)
        f.write("# (EleutherAI/gpt-neo-125m @ 21def018). Format: escaped text <TAB> ids <TAB>\n")
        f.write("# pre-token byte lengths (HF ByteLevel pre-tokenizer).\n")
        f.write("# Escapes: \\\\ \\n \\t \\r \\xHH (controls, and a leading '#').\n")
        f.write("\n".join(rows) + "\n")
    with open(OUT_VOCAB, "w", encoding="ascii", newline="\n") as f:
        json.dump(subset, f, ensure_ascii=True)  # \\uXXXX escapes exercise the JSON reader
    with open(OUT_MERGES, "w", encoding="utf-8", newline="\n") as f:
        f.write("#version: 0.2 - subset of GPT-Neo merges.txt (original order), generated\n")
        f.write("\n".join(merge_lines[r] for r in used_sorted) + "\n")
    print(f"{len(rows)} cases, {len(used_sorted)} merges, {len(subset)} vocab entries")


if __name__ == "__main__":
    main()
