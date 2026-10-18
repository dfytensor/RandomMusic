# -*- coding: utf-8 -*-
"""ABC vocal-melody parser + DTW similarity between original and cover transcriptions."""
import re

import numpy as np

LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]
SIG_SHARPS = {"G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7}
SIG_FLATS = {"F": 1, "Bb": 2, "Eb": 3, "Ab": 4, "Db": 5, "Gb": 6, "Cb": 7}
TOKEN = re.compile(r"(?P<acc>[_^=]?)(?P<let>[A-Ga-gzZ])(?P<oct>[,']*)(?P<len>\d+)?(?P<frac>/\d+)?(?P<tie>-)?")
NOTE_TOKEN = re.compile(r"[_^=]?[A-Ga-gzZ][,']*\d*(?:/\d+)?-?")
SECTION = re.compile(r"^\s*%\s*(\S+)\s*$")


def keysig_alterations(kline):
    m = re.match(r"([A-G])(#|b)?", kline)
    if not m:
        return {}
    tonic, acc = m.group(1), m.group(2) or ""
    if acc == "#":
        n = SIG_SHARPS.get(tonic + "#", SIG_SHARPS.get(tonic, 0))
        return {L: 1 for L in SHARP_ORDER[:n]}
    if acc == "b":
        name = tonic + "b" if tonic + "b" in SIG_FLATS else tonic
        n = SIG_FLATS.get(name, 0)
        return {L: -1 for L in FLAT_ORDER[:n]}
    return {}


def parse_abc(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    keysig = {}
    q_per_quarter = 60.0
    unit_quarter = 0.25      # L:1/16 default: one unit = 1/16 = 0.25 quarter
    beats_per_bar = 4.0      # M:4/4 default
    beat_den = 4
    voice = "Vocal"
    pos = 0.0
    notes = []
    sections = []
    pending = None

    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("%%"):
            continue
        if s.startswith("%"):
            m = SECTION.match(s)
            if m:
                sections.append((m.group(1), pos))
            continue
        if s.startswith("Q:"):
            qm = re.search(r"=([0-9.]+)", s)
            if qm:
                q_per_quarter = float(qm.group(1))
            continue
        if s.startswith("L:"):
            lm = re.match(r"L:1/(\d+)", s)
            if lm:
                unit_quarter = 4.0 / int(lm.group(1))
            continue
        if s.startswith("K:"):
            keysig = keysig_alterations(s[2:].strip())
            continue
        if s.startswith("M:"):
            mm = re.match(r"M:(\d+)/(\d+)", s)
            if mm:
                beats_per_bar = float(mm.group(1))
                beat_den = int(mm.group(2))
            continue
        if s.startswith("V:"):
            if "Ins" in s:
                voice = "Ins"
            elif "Vocal" in s:
                voice = "Vocal"
            continue
        if s.startswith(("X:", "T:", "C:")):
            continue

        bar_units = beats_per_bar * (4.0 / beat_den) * unit_quarter
        s_music = re.sub(r'"[^"]*"', "", s)  # strip chord symbols
        for tok in NOTE_TOKEN.findall(s_music):
            if tok in ("|", "||", "|]"):
                continue
            if voice != "Vocal":
                continue
            m = TOKEN.fullmatch(tok)
            if not m:
                continue
            let = m.group("let")
            n = int(m.group("len")) if m.group("len") else 1
            frac = 1.0 / int(m.group("frac")[1:]) if m.group("frac") else 1.0
            if let in ("z", "Z"):
                u = n * bar_units if let == "Z" else n * frac
                if pending is not None:
                    notes.append(pending)
                    pending = None
                pos += u
                continue
            acc = m.group("acc")
            octaves = m.group("oct").count("'") - m.group("oct").count(",")
            midi = 60 + LETTER[let.upper()] + 12 * octaves
            if let.islower():
                midi += 12
            if acc == "^":
                midi += 1
            elif acc == "_":
                midi -= 1
            elif acc == "":
                midi += keysig.get(let.upper(), 0)
            u = n * frac
            if m.group("tie"):
                if pending is not None and pending[1] == midi:
                    pending = (pending[0], midi, pending[2] + u)
                else:
                    if pending is not None:
                        notes.append(pending)
                    pending = (pos, midi, u)
                pos += u
            else:
                if pending is not None:
                    notes.append(pending)
                    pending = None
                notes.append((pos, midi, u))
                pos += u
    if pending is not None:
        notes.append(pending)

    sec_per_unit = unit_quarter * 60.0 / q_per_quarter
    vocal = [(p * sec_per_unit, mi, d * sec_per_unit) for p, mi, d in notes]
    secs = [(name, p * sec_per_unit) for name, p in sections]
    sec_per_bar = beats_per_bar * (4.0 / beat_den) * 60.0 / q_per_quarter
    return dict(q=q_per_quarter, total_sec=pos * sec_per_unit, sec_per_bar=sec_per_bar), vocal, secs


def to_grid(vocal, total_sec, rate=8.0):
    n = max(1, int(total_sec * rate))
    grid = np.zeros(n)
    for onset, midi, dur in vocal:
        a, b = int(onset * rate), min(n, int((onset + dur) * rate))
        if b > a:
            grid[a:b] = midi
    return grid


def dtw_similarity(a, b, band_rad=0.15):
    from librosa.sequence import dtw
    x = np.linspace(0, 1, len(b))
    xi = np.linspace(0, 1, len(a))
    b_rs = np.interp(xi, x, b)
    D, wp = dtw(a.reshape(1, -1), b_rs.reshape(1, -1), metric="euclidean", band_rad=band_rad)
    diffs = []
    both = rest_both = 0
    for i, j in wp:
        va, vb = a[i], b_rs[j]
        if va > 0 and vb > 0:
            diffs.append(abs(va - vb))
            both += 1
        elif (va > 0) != (vb > 0):
            diffs.append(12.0)
        else:
            rest_both += 1
    mad = float(np.mean(diffs)) if diffs else 99.0
    pitch_acc = round(float(1 - np.mean(diffs[:both]) / 12), 3) if both else 0.0
    return dict(mean_abs_semitones=round(mad, 2),
                note_frame_similarity=round(max(0.0, 1.0 - mad / 12.0), 3),
                pitch_acc_on_both_notes=pitch_acc,
                both_note_frames=both, rest_frames=rest_both,
                dtw_norm_dist=round(float(D[-1, -1] / len(wp)), 2))


def pitch_hist(vocal):
    h = np.zeros(12)
    for _, midi, dur in vocal:
        h[int(midi) % 12] += dur
    return h / (h.sum() + 1e-9)


def hist_cosine(h1, h2):
    return round(float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-9)), 3)


def transpose_min_mad(a, b, lo=-3, hi=3):
    """Try relative transpositions of cover to original; return best."""
    best = None
    for t in range(lo, hi + 1):
        r = dtw_similarity(a, b + t)
        r["transpose"] = t
        if best is None or r["mean_abs_semitones"] < best["mean_abs_semitones"]:
            best = r
    return best
