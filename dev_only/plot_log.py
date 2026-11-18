"""Plot train/val loss curves from one or more tslm log.csv files as an SVG.

Offline only (dev_only/), standard library only (no plotting dependency).
Reads the frozen log.csv columns (ROADMAP §9): train_loss on every logged
row, val_loss on validation rows. The x axis is tokens_seen (millions).

    python3 dev_only/plot_log.py <out.svg> <run>/log.csv [more log.csv ...]

Each file is drawn as a solid train line and a dashed val line with dots,
labelled by its parent directory name. A summary (last train and val
values, minimum val) is printed to stdout.
"""
import csv
import math
import os
import sys

W, H = 900, 540
ML, MR, MT, MB = 70, 220, 30, 60
COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]


def read_log(path):
    train, val = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            x = int(row["tokens_seen"]) / 1e6
            train.append((x, float(row["train_loss"])))
            if row["val_loss"]:
                val.append((x, float(row["val_loss"])))
    if not train:
        sys.exit(f"{path}: no rows")
    return train, val


def nice_ticks(lo, hi, n=6):
    span = hi - lo or 1.0
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.floor(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            out.append(round(t, 10))
        t += step
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out, paths = sys.argv[1], sys.argv[2:]
    runs = [(os.path.basename(os.path.dirname(os.path.abspath(p))), *read_log(p)) for p in paths]
    xs = [x for _, tr, va in runs for x, _ in tr + va]
    ys = [y for _, tr, va in runs for _, y in tr + va]
    x0, x1 = 0.0, max(xs)
    y0, y1 = math.floor(min(ys) * 2) / 2, math.ceil(max(ys) * 2) / 2
    sx = lambda x: ML + (x - x0) / ((x1 - x0) or 1) * (W - ML - MR)
    sy = lambda y: MT + (y1 - y) / ((y1 - y0) or 1) * (H - MT - MB)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    for t in nice_ticks(y0, y1):
        s.append(f'<line x1="{ML}" x2="{W-MR}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="#e5e7eb"/>')
        s.append(f'<text x="{ML-8}" y="{sy(t)+4:.1f}" text-anchor="end">{t:g}</text>')
    for t in nice_ticks(x0, x1):
        s.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{MT}" y2="{H-MB}" stroke="#f3f4f6"/>')
        s.append(f'<text x="{sx(t):.1f}" y="{H-MB+18}" text-anchor="middle">{t:g}</text>')
    s.append(f'<rect x="{ML}" y="{MT}" width="{W-ML-MR}" height="{H-MT-MB}" fill="none" stroke="#9ca3af"/>')
    s.append(f'<text x="{(ML+W-MR)/2}" y="{H-18}" text-anchor="middle">tokens seen (millions)</text>')
    s.append(f'<text x="18" y="{(MT+H-MB)/2}" text-anchor="middle" transform="rotate(-90 18 {(MT+H-MB)/2})">'
             f'cross-entropy loss (nats)</text>')
    for i, (name, tr, va) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in tr)
        s.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.5" opacity="0.7"/>')
        if va:
            pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in va)
            s.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2" stroke-dasharray="6 4"/>')
            for x, y in va:
                s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{c}"/>')
        ly = MT + 10 + 44 * i
        s.append(f'<line x1="{W-MR+15}" x2="{W-MR+45}" y1="{ly}" y2="{ly}" stroke="{c}" stroke-width="1.5"/>')
        s.append(f'<text x="{W-MR+52}" y="{ly+4}">{name} train</text>')
        s.append(f'<line x1="{W-MR+15}" x2="{W-MR+45}" y1="{ly+18}" y2="{ly+18}" stroke="{c}" stroke-width="2" '
                 f'stroke-dasharray="6 4"/>')
        s.append(f'<text x="{W-MR+52}" y="{ly+22}">{name} val</text>')
        last_v = f"{va[-1][1]:.4f}" if va else "n/a"
        min_v = f"{min(v for _, v in va):.4f}" if va else "n/a"
        print(f"{name}: {len(tr)} rows, {tr[-1][0]:.2f}M tokens, last train {tr[-1][1]:.4f}, "
              f"last val {last_v}, min val {min_v}")
    s.append("</svg>")
    with open(out, "w") as f:
        f.write("\n".join(s) + "\n")


if __name__ == "__main__":
    main()
