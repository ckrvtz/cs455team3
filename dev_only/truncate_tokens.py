"""Write the first N tokens of a TSTK token file (ROADMAP §7.3) as a new token file.

Offline only (dev_only/): used to make the committed smoke-test validation
fixture tests/fixtures/smoke_val.bin from the official val.bin. The header is
copied with num_tokens replaced; tokens are copied byte for byte.

    python3 dev_only/truncate_tokens.py <in.bin> <out.bin> <num_tokens>
"""
import struct
import sys


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    data = open(src, "rb").read()
    if data[:4] != b"TSTK" or len(data) < 32:
        sys.exit(f"{src}: not a token file")
    have = struct.unpack_from("<Q", data, 16)[0]
    if not 1 <= n <= have:
        sys.exit(f"{src}: has {have} tokens, asked for {n}")
    header = bytearray(data[:32])
    struct.pack_into("<Q", header, 16, n)
    with open(dst, "wb") as f:
        f.write(bytes(header) + data[32:32 + 2 * n])


if __name__ == "__main__":
    main()
