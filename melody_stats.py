# -*- coding: utf-8 -*-
"""Melody & structure statistics from corpus transcriptions -> reference ranges for the map."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from abc_compare import parse_abc

ROOT = Path(__file__).parent
SKIP_SECTIONS = {"intro", "interlude", "outro"}
QUARTER_UNITS = 4.0  # L:1/16


def section_bars(meta, secs):
    out = []
    for i, (name, t) in enumerate(secs):
        t2 = secs[i + 1][1] if i + 1 < len(secs) else meta["total_sec"]
        bars = (t2 - t) / meta["sec_per_bar"]
        out.append((name, bars))
    return out


def notes_in(vocal, t1, t2):
    return [(o, m, d) for o, m, d in vocal if t1 <= o < t2]


def three_gram_repeat(pitches):
    if len(pitches) < 6:
        return None
    grams = [tuple(pitches[i:i + 3]) for i in range(len(pitches) - 2)]
    return 1.0 - len(set(grams)) / len(grams)


def analyze_song(abc_path):
    abc_path = Path(abc_path)
    meta, vocal, secs = parse_abc(str(abc_path))
    if len(vocal) < 20 or not secs:
        return None
    sbars = section_bars(meta, secs)
    total_bars = sum(b for _, b in sbars)
    times = [t for _, t in secs] + [meta["total_sec"]]

    dens = len(vocal) / max(total_bars, 1)
    pitches = [m for _, m, _ in vocal]
    rng = max(pitches) - min(pitches)
    p10, p90 = np.percentile(pitches, [10, 90])

    steps = leaps = 0
    for a, b in zip(pitches, pitches[1:]):
        iv = abs(b - a)
        if iv == 0:
            continue
        if iv <= 2:
            steps += 1
        elif iv >= 5:
            leaps += 1
    leap_ratio = leaps / max(steps + leaps, 1)

    # motif repetition per melodic section, averaged
    reps = []
    for i, (name, t) in enumerate(secs):
        if name.lower() in SKIP_SECTIONS:
            continue
        t2 = times[i + 1]
        ns = notes_in(vocal, t, t2)
        r = three_gram_repeat([m for _, m, _ in ns])
        if r is not None:
            reps.append(r)
    rep = float(np.mean(reps)) if reps else None

    # chorus lift vs verse
    def first(name):
        for i, (n, t) in enumerate(secs):
            if n.lower() == name:
                return times[i], times[i + 1]
        return None
    lift_mean = lift_max = dens_ratio = None
    fv, fc = first("verse"), first("chorus")
    if fv and fc:
        nv = notes_in(vocal, *fv)
        nc = notes_in(vocal, *fc)
        if len(nv) >= 8 and len(nc) >= 8:
            mp_v = [m for _, m, _ in nv]
            mp_c = [m for _, m, _ in nc]
            lift_mean = float(np.mean(mp_c) - np.mean(mp_v))
            lift_max = float(max(mp_c) - max(mp_v))
            spb = meta["sec_per_bar"]
            bv = (fv[1] - fv[0]) / spb
            bc = (fc[1] - fc[0]) / spb
            dens_ratio = (len(nc) / bc) / (len(nv) / bv)

    rest_ratio = None
    if vocal:
        t_end = meta["total_sec"]
        sounding = sum(d for _, _, d in vocal)
        rest_ratio = 1.0 - sounding / t_end

    chain = "->".join(n.lower() for n, _ in secs[:4])
    return dict(file=abc_path.parent.name, chain=chain, total_bars=round(total_bars, 1),
                density=round(dens, 2), pitch_range=rng, tessitura=[int(p10), int(p90)],
                leap_ratio=round(leap_ratio, 3),
                motif_repeat=round(rep, 3) if rep is not None else None,
                lift_mean=round(lift_mean, 1) if lift_mean is not None else None,
                lift_max=round(lift_max, 1) if lift_max is not None else None,
                dens_ratio=round(dens_ratio, 2) if dens_ratio is not None else None,
                rest_ratio=round(rest_ratio, 3) if rest_ratio is not None else None)


def stats(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0:
        return None
    return dict(n=len(a), p25=round(float(np.percentile(a, 25)), 2), med=round(float(np.median(a)), 2),
                p75=round(float(np.percentile(a, 75)), 2))


def main():
    files = sorted((ROOT / "corpus" / "transcriptions").glob("*/score.abc"))
    rows, chains, sec_bars = [], Counter(), {}
    for f in files:
        try:
            r = analyze_song(f)
        except Exception as e:
            print(f"parse fail {f.parent.name}: {str(e)[:80]}")
            continue
        if r:
            rows.append(r)
            chains[r["chain"]] += 1
            meta, vocal, secs = parse_abc(str(f))
            for name, bars in section_bars(meta, secs):
                sec_bars.setdefault(name.lower(), []).append(bars)

    agg = {
        "songs": len(rows),
        "density_notes_per_bar": stats([r["density"] for r in rows]),
        "pitch_range_semitones": stats([r["pitch_range"] for r in rows]),
        "leap_ratio": stats([r["leap_ratio"] for r in rows]),
        "motif_repeat": stats([r["motif_repeat"] for r in rows]),
        "chorus_lift_mean_semitones": stats([r["lift_mean"] for r in rows]),
        "chorus_lift_max_semitones": stats([r["lift_max"] for r in rows]),
        "chorus_density_ratio": stats([r["dens_ratio"] for r in rows]),
        "rest_ratio": stats([r["rest_ratio"] for r in rows]),
        "front_structure_chains": chains.most_common(8),
        "section_bars": {k: stats(v) for k, v in sorted(sec_bars.items())},
    }
    (ROOT / "corpus" / "melody_stats.json").write_text(
        json.dumps(dict(aggregate=agg, per_song=rows), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
